"""MCP view of the same workspace (D1, D3): publishes TOOLS unchanged and dispatches
through call_tool, so MCP clients and env.step see identical schemas, results, and errors.

One server process = one episode's world. There is deliberately no reset tool:
a reset the agent can call is a hack vector (undo mistakes, probe, retry)."""
import json

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from opsgym.seeds import SSO_SEED
from opsgym.tools import TOOLS, call_tool
from opsgym.workspace import load_seed


def build_server(seed: dict) -> Server:
    conn = load_seed(seed)
    server = Server("opsgym")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [types.Tool(name=name, description=t["description"], inputSchema=t["input_schema"])
                for name, t in TOOLS.items()]

    # validate_input=False: call_tool already validates against the same schema. If the SDK
    # validated too, MCP clients would get the SDK's error text instead of ours: two paths.
    @server.call_tool(validate_input=False)
    async def handle(name: str, arguments: dict) -> types.CallToolResult:
        obs = call_tool(conn, name, arguments)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(obs))],
            structuredContent=obs, isError="error" in obs)

    return server


async def _main() -> None:
    # ponytail: always serves SSO_SEED; add a --seed path arg when Phase 4 has more seeds.
    server = build_server(SSO_SEED)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    anyio.run(_main)
