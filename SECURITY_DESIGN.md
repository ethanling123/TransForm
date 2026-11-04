# Secure data plane: threat model and plan

Everything here is behind `TRANSFORM_SECURE=1`. With it off the server behaves exactly as before, so the "before" attack
results, the unit tests and the golden digests stay valid and before/after is a like-for-like comparison.

## Threat model

Assets: result payloads (may be sensitive tables), the server's credentials, and the model's context.

Attackers:

- **A co-tenant** holding valid credentials of their own (T1, T7).
- **A network attacker** on the data-plane link between client and server (T5), or anyone who obtains a fetch link (T2, T3).
- **Content in the data** the agent reads: text or cells written by a third party (T9), and a model that is steered into
  emitting a link (T3).
- **A forged or edited descriptor** reaching the client (T4, T6).

Trusted: the server process, the client library (`client/mcp_client.py`), and the signing key held by the server.
Not trusted: the model, payload contents, the network, other principals.

Out of scope: TLS itself (it protects the link, not descriptors, links, tenants or writes; the paper should still measure its
cost next to ours), MCP OAuth (a real deployment would supply the principal from it; here principals come from API keys),
a compromised server, and side channels.

## Defences and the attacks they address

| Step | Defence | Attacks |
|---|---|---|
| 1 | Principals: each key maps to a principal; every result handle has an owner; every route and tool checks it | T1, T7 |
| 2 | Fetch URLs carry `exp` and an HMAC over handle, arm, principal and expiry (path only, not the host) | T2, T3 |
| 3 | Descriptors are signed (Ed25519), cover the requested `result_id`, and carry the payload sha256; the client verifies both | T4, T5 |
| 4 | The client only follows URLs on the MCP server's origin and never sends the bearer key elsewhere | T6 |
| 5 | Inline text and cells from untrusted data are datamarked inside an envelope; measured against a real model | T9 |

Step 3 details: a grant is `{source, id, arm, path, digest, digest_of, principal, exp}` signed with Ed25519. `source` is the
result id the tool call was about, and the client refuses a grant issued for any other result (a genuine descriptor swapped in
for another result still carries a valid signature). Raw byte arms share one digest of the original bytes, hashed once per file
and checked after the client decompresses, so no codec runs at mint time. Every other blob arm is hashed over the response
body, which may need the payload encoded at mint time when registration has not already done it; that is the cost the
security-aware selector has to account for. Streams have no single body: they carry no digest and the verifying client
refuses them. The client pins the public key from the `signing_public_key` tool.

Step 4 details: `client.mcp_client.connect(strict_origins=...)`, on by default when `TRANSFORM_SECURE=1`, makes the returned
HTTP client raise `ForeignOriginError` before sending any request, and so any bearer key, to an origin other than the MCP
server's. A loopback server also allows the other loopback spellings on its port (the server writes `localhost` into the URLs it
mints). When the server is reached by another name than the one in its URLs (the Docker setup), list the URL origin in
`TRANSFORM_ALLOWED_ORIGINS`. The client library is part of the trusted base; a host that fetches URLs without it is not covered.

Expiry only bounds replay; it does not stop a link used within its lifetime. T1, T3 and T7 are only meaningful with
authentication, so open mode reports them as "requires auth", not as failures.

T1 cannot be shown with one shared key. The defended run adds a second principal with a different key as a documented
variant; the original "before" attack is unchanged.

## Measuring

`benchmarks/attacks/run_attacks.py --label defended` after each step. Then per-arm latency and byte overhead with defences
on versus off, so the cost can feed `cost(arm)` in the selector (security-aware selection).

## Deployment notes

`TRANSFORM_URL_SECRET` (and the descriptor signing key from step 3) default to a random value per process. Set them
explicitly for multi-worker or two-machine runs, or links minted by one process are rejected by another.

Steps 1 and 2 are the baseline that object stores already use (per-tenant access control, presigned URLs). The
contribution is the part specific to MCP and agents: descriptor integrity, keeping links out of the model's context,
untrusted-text handling measured on a real model, and security-aware format selection.
