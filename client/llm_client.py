"""
Lightweight LLM client for the workflow.

Supports free / low-cost options as requested:
- Ollama (local Llama): free, no API key. Run `ollama run llama3.2` (or similar) first.
- Deep Seek: set DEEPSEEK_API_KEY for api.deepseek.com (low-cost, often free-tier offers).
- Vertex AI (Gemini), two auth modes:
  * Express mode / Agent Platform: set VERTEX_API_KEY. No project, no google-auth.
  * Standard Vertex: set VERTEX_PROJECT_ID + GCP credentials (ADC or a service
    account via GOOGLE_APPLICATION_CREDENTIALS). Requires `pip install google-auth`.
  VERTEX_API_KEY takes precedence when both are present.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Literal, Optional

import httpx

# Backend selection: "ollama" (Llama via Ollama), "deepseek", or "vertex" (GCP Vertex AI Gemini)
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
VERTEX_PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
# Express-mode / Agent Platform API key. When set, used instead of OAuth (ADC).
VERTEX_API_KEY = os.environ.get("VERTEX_API_KEY", "")


def get_llm_backend() -> Literal["ollama", "deepseek", "vertex"]:
    if LLM_BACKEND in ("ollama", "deepseek", "vertex"):
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
    api_key: Optional[str] = None,
    timeout: float = 120.0,
) -> str:
    """
    Call Vertex AI's generateContent REST endpoint for a Gemini model.

    Two auth modes, API key first:

    - Express mode / Agent Platform: VERTEX_API_KEY (or api_key=). Hits the
      global aiplatform endpoint, which takes no project or location.
    - Standard Vertex: VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) plus ADC
      (`gcloud auth application-default login`) or a service account key in
      GOOGLE_APPLICATION_CREDENTIALS. Requires `pip install google-auth`.

    Note that model availability differs between the two: express-mode
    projects may expose only a subset (e.g. gemini-2.5-flash / -pro).
    """
    contents, system_instruction = _messages_to_vertex_contents(messages)
    body: Dict[str, Any] = {"contents": contents}
    if system_instruction:
        body["systemInstruction"] = system_instruction

    key = api_key or VERTEX_API_KEY
    if key:
        url = (
            "https://aiplatform.googleapis.com/v1/publishers/google/models/"
            f"{model}:generateContent"
        )
        params: Dict[str, str] = {"key": key}
        headers = {"Content-Type": "application/json"}
    else:
        project = project_id or VERTEX_PROJECT_ID
        if not project:
            raise ValueError(
                "No Vertex credentials. Set VERTEX_API_KEY for express mode, or "
                "VERTEX_PROJECT_ID (or GOOGLE_CLOUD_PROJECT) for standard Vertex."
            )
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as e:
            raise RuntimeError(
                "google-auth is required for OAuth-based Vertex access: "
                "pip install google-auth"
            ) from e

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds.refresh(google.auth.transport.requests.Request())
        url = (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{location}/publishers/google/models/{model}:generateContent"
        )
        params = {}
        headers = {
            "Authorization": f"Bearer {creds.token}",
            "Content-Type": "application/json",
        }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(url, headers=headers, params=params, json=body)
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
    backend: Optional[Literal["ollama", "deepseek", "vertex"]] = None,
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
