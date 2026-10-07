"""Diff the MCP tool names and argument schemas of two running TransForm servers."""

from __future__ import annotations

import argparse
import asyncio
import sys

from client.mcp_client import connect


async def tools(port: int):
    async with connect(f"http://localhost:{port}/mcp/mcp") as (session, _):
        result = await session.list_tools()
    return {
        t.name: {
            "props": sorted((t.inputSchema or {}).get("properties", {})),
            "required": sorted((t.inputSchema or {}).get("required", [])),
        }
        for t in result.tools
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=int, required=True, help="port of server A (e.g. python)")
    parser.add_argument("--b", type=int, required=True, help="port of server B (e.g. java)")
    args = parser.parse_args()
    a, b = await tools(args.a), await tools(args.b)
    print(f"server A: {len(a)} tools | server B: {len(b)} tools")
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    diffs = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    print(f"only in A: {only_a}\nonly in B: {only_b}\nschema differences: {diffs}")
    sys.exit(1 if (only_a or only_b or diffs) else 0)


if __name__ == "__main__":
    asyncio.run(main())
