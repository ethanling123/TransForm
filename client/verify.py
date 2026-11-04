"""Verify a signed descriptor grant and the bytes behind it (server side: python-server/security.py, TRANSFORM_SECURE=1).

The client pins the server's Ed25519 public key (`trusted_public_key`, or pass one in), then `fetch_verified` refuses to return
bytes unless the grant is signed by that key, was issued for the result the caller asked about, points at the URL being
fetched, and the fetched bytes hash to the signed digest.
"""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

import httpx
import pyarrow as pa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from mcp import ClientSession

from client.mcp_client import _unwrap_structured_content

RAW_CODECS = {"zstd_blob": "zstd", "brotli_blob": "brotli"}


class IntegrityError(Exception):
    pass


def find_grant(out: Dict[str, Any]) -> Dict[str, Any]:
    grant = out.get("grant") or (out.get("payload") or {}).get("grant")
    if not grant:
        raise IntegrityError("descriptor carries no signed grant")
    return grant


def verify_grant(grant: Dict[str, Any], public_key: str, expected_source: str) -> None:
    body = {k: v for k, v in grant.items() if k != "sig"}
    try:
        Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key)).verify(
            base64.b64decode(grant.get("sig", "")), json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    except (InvalidSignature, ValueError):
        raise IntegrityError("grant signature does not verify") from None
    if grant["source"] != expected_source:
        raise IntegrityError(f"grant was issued for result {grant['source']}, not the {expected_source} that was asked for")


async def trusted_public_key(session: ClientSession) -> str:
    result = await session.call_tool("signing_public_key", arguments={})
    key = (_unwrap_structured_content(result.structuredContent) or {}).get("public_key")
    if not key:
        raise IntegrityError("server does not sign descriptors (TRANSFORM_SECURE is off)")
    return key


async def fetch_verified(client: httpx.AsyncClient, out: Dict[str, Any], *, expected_source: str,
                         public_key: Optional[str] = None, session: Optional[ClientSession] = None) -> bytes:
    """Fetch the blob a descriptor points at and return its verified bytes.

    `out` is a tool result's structured content (large_result_auto, large_parquet_blob, fetch_arm, ...). For raw byte arms the
    returned bytes are the original payload; for table arms they are the response body, ready to decode.
    """
    key = public_key or await trusted_public_key(session)
    grant = find_grant(out)
    verify_grant(grant, key, expected_source)
    url = (out.get("decode") or {}).get("url") or out.get("url") or (out.get("payload") or {}).get("url")
    if not url or urlsplit(url).path != grant["path"]:
        raise IntegrityError(f"descriptor URL {url!r} is not the path the grant covers ({grant['path']})")
    if grant["digest"] is None:
        raise IntegrityError(f"arm {grant['arm']} has no digest to check (streams are not covered)")
    resp = await client.get(url)
    resp.raise_for_status()
    body = resp.content
    if grant["digest_of"] == "raw" and grant["arm"] in RAW_CODECS:
        size = int(resp.headers.get("x-uncompressed-bytes") or (out.get("decode") or {}).get("uncompressed_bytes") or 0)
        try:
            body = pa.Codec(RAW_CODECS[grant["arm"]]).decompress(body, decompressed_size=size, asbytes=True)
        except pa.ArrowException as exc:
            raise IntegrityError(f"payload does not decompress: {exc}") from None
    if hashlib.sha256(body).hexdigest() != grant["digest"]:
        raise IntegrityError("fetched bytes do not match the signed digest")
    return body
