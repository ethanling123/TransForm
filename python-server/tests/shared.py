"""One TestClient per process: the MCP session manager can only be started once, so test modules share it."""

from starlette.testclient import TestClient

_client = None


def client(app) -> TestClient:
    global _client
    if _client is None:
        _client = TestClient(app)
        _client.__enter__()
    return _client
