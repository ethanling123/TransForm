"""Secure data plane (off unless TRANSFORM_SECURE=1). See SECURITY_DESIGN.md at the repo root.

Step 1: principals. MCP_API_KEYS="alice:key1,bob:key2" maps each bearer key to a principal; MCP_API_KEY alone is the single
principal "default"; with neither set every caller is "anonymous". Every result handle records the principal that
created it, and a handle is invisible (indistinguishable from an unknown id) to every other principal.
"""

from __future__ import annotations

import contextvars
import hmac
import os
from typing import Any, Dict, Optional

ANONYMOUS = "anonymous"
current_principal: contextvars.ContextVar[str] = contextvars.ContextVar("transform_principal", default=ANONYMOUS)


def enabled() -> bool:
    return os.environ.get("TRANSFORM_SECURE", "").strip().lower() in ("1", "true", "yes")


def load_keys() -> Dict[str, str]:
    """Bearer key -> principal."""
    keys: Dict[str, str] = {}
    for item in os.environ.get("MCP_API_KEYS", "").split(","):
        name, sep, key = item.strip().partition(":")
        if sep and name and key:
            keys[key] = name
    single = os.environ.get("MCP_API_KEY", "").strip()
    if single and single not in keys:
        keys[single] = "default"
    return keys


def principal_for(token: str, keys: Dict[str, str]) -> Optional[str]:
    found = None
    for key, name in keys.items():
        if hmac.compare_digest(token.encode(), key.encode()):
            found = name
    return found


class OwnedRegistry(dict):
    """Result registry that stamps an owner on every entry and hides entries from other principals.

    With the flag off it is a plain dict. `raw` bypasses the check for server-internal bookkeeping such as cache eviction.
    """

    def raw(self, key: str) -> Optional[Any]:
        return dict.get(self, key)

    def _visible(self, cfg: Any) -> bool:
        return cfg is not None and (not enabled() or getattr(cfg, "owner", None) == current_principal.get())

    def get(self, key, default=None):
        cfg = dict.get(self, key)
        return cfg if self._visible(cfg) else default

    def __getitem__(self, key):
        cfg = dict.get(self, key)
        if not self._visible(cfg):
            raise KeyError(key)
        return cfg

    def __contains__(self, key):
        return self._visible(dict.get(self, key))

    def __setitem__(self, key, cfg):
        if enabled() and getattr(cfg, "owner", None) is None:
            cfg.owner = current_principal.get()
        dict.__setitem__(self, key, cfg)

