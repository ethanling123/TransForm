"""
Lightweight LLM client for the workflow.

Supports free / low-cost options as requested:
- Ollama (local Llama): free, no API key. Run `ollama run llama3.2` (or similar) first.
- Deep Seek: set DEEPSEEK_API_KEY for api.deepseek.com (low-cost, often free-tier offers).
- Vertex AI (Gemini): set VERTEX_PROJECT_ID + GCP credentials (ADC or a service
  account via GOOGLE_APPLICATION_CREDENTIALS). Requires `pip install google-auth`.
- Gemini API (AI Studio): set GEMINI_API_KEY (from https://aistudio.google.com/apikey).
  No GCP project/billing setup required.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import httpx

# Backend selection: "ollama" (Llama via Ollama), "deepseek", "vertex" (GCP Vertex AI
# Gemini), or "gemini" (AI Studio Gemini API, no GCP project/billing required)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.0-flash-001")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")


def get_llm_backend() -> Literal["ollama", "deepseek", "vertex", "gemini"]:
    if LLM_BACKEND in ("ollama", "deepseek", "vertex", "gemini"):
        return LLM_BACKEND  # type: ignore
    return "ollama"


def _messages_to_ollama_prompt(messages: List[Dict[str, str]]) -> str:
    """Convert OpenAI-style messages to a single prompt for Ollama."""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    if parts:
        parts.append("Assistant:")
    return "\n\n".join(parts)


async def chat_ollama(
    messages: List[Dict[str, str]],
    *,
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: float = 120.0,
) -> str:
    """Call Ollama /api/chat (Llama or other local model)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
    return (data.get("message") or {}).get("content", "").strip()


async def chat_deepseek(
    messages: List[Dict[str, str]],
    *,
    model: str = DEEPSEEK_MODEL,
    base_url: str = DEEPSEEK_BASE_URL,
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """Call Deep Seek API (OpenAI-compatible). Requires DEEPSEEK_API_KEY."""
    key = api_key or DEEPSEEK_API_KEY
    if not key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Set it in the environment or pass api_key=..."
        )
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "stream": False},
        )
        r.raise_for_status()
        data = r.json()
    choice = (data.get("choices") or [None])[0]
    if not choice:
        return ""
    return (choice.get("message") or {}).get("content", "").strip()


def _messages_to_vertex_contents(
    messages: List[Dict[str, str]]
) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convert OpenAI-style messages to Vertex AI generateContent contents +
    systemInstruction."""
    contents: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "assistant":
            contents.append({"role": "model", "parts": [{"text": content}]})
        else:
            contents.append({"role": "user", "parts": [{"text": content}]})
    system_instruction = (
        {"parts": [{"text": "\n\n".join(system_parts)}]} if system_parts else None
    )
    return contents, system_instruction


async def chat_vertex(
    messages: List[Dict[str, str]],
    *,
    model: str = VERTEX_MODEL,
    project_id: Optional[str] = None,
    location: str = VERTEX_LOCATION,
    timeout: float = 120.0,
) -> str:
    """
    Call Vertex AI's generateContent REST endpoint for a Gemini model.

    Requires `pip install google-auth` and GCP credentials: either
    Application Default Credentials (`gcloud auth application-default login`)
    or a service account key referenced by GOOGLE_APPLICATION_CREDENTIALS.
    Requires VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) to be set.
    """
    project = project_id or VERTEX_PROJECT_ID
    if not project:
        raise ValueError(
            "VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) is not set. "
            "Set it to your GCP project ID."
        )
    try:
        import google.auth
        import google.auth.transport.requests
    except ImportError as e:
        raise RuntimeError(
            "google-auth is required for the vertex backend: pip install google-auth"
        ) from e

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())

    contents, system_instruction = _messages_to_vertex_contents(messages)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
        f"/locations/{location}/publishers/google/models/{model}:generateContent"
    )
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {creds.token}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


async def chat_gemini(
    messages: List[Dict[str, str]],
    *,
    model: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
    base_url: str = GEMINI_BASE_URL,
    timeout: float = 120.0,
) -> str:
    """
    Call the AI Studio Gemini API's generateContent endpoint.

    Requires GEMINI_API_KEY (from https://aistudio.google.com/apikey). No GCP
    project or billing setup needed.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Set it in the environment or pass api_key=..."
        )
    contents, system_instruction = _messages_to_vertex_contents(messages)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    url = f"{base_url.rstrip('/')}/v1beta/models/{model}:generateContent"
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(
            url,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


async def chat(
    messages: List[Dict[str, str]],
    *,
    backend: Optional[Literal["ollama", "deepseek", "vertex", "gemini"]] = None,
    **kwargs: Any,
) -> str:
    """
    Single entry point for the lightweight LLM.
    Uses LLM_BACKEND env (default ollama). Pass backend= to override.
    """
    backend = backend or get_llm_backend()
    if backend == "ollama":
        return await chat_ollama(messages, **kwargs)
    if backend == "deepseek":
        return await chat_deepseek(messages, **kwargs)
    if backend == "vertex":
        return await chat_vertex(messages, **kwargs)
    if backend == "gemini":
        return await chat_gemini(messages, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")


async def complete(prompt: str, **kwargs: Any) -> str:
    """Convenience: single user prompt."""
    return await chat([{"role": "user", "content": prompt}], **kwargs)


if __name__ == "__main__":
    import asyncio
    import sys

    prompt = " ".join(sys.argv[1:]) or "Say hello in one short sentence."
    backend = get_llm_backend()
    print(f"backend={backend}", file=sys.stderr)
    print(asyncio.run(complete(prompt)))
