"""End-to-end check of the real deployment path: an MCP client on the host talks over
stdio to the server inside the locked-down container. Run after `docker build -t opsgym .`:
    python scripts/smoke_mcp_docker.py
"""
import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER = StdioServerParameters(
    command="docker", args=["run", "-i", "--rm", "--network", "none", "--read-only", "opsgym"])


async def main():
    async with stdio_client(SERVER) as (read, write), ClientSession(read, write) as client:
        await client.initialize()
        names = sorted(t.name for t in (await client.list_tools()).tools)
        assert names == ["create_ticket", "get_account", "list_tickets", "search_accounts", "search_messages"], names
        acme = await client.call_tool("get_account", {"account_id": "acc_acme"})
        assert acme.structuredContent["owner_id"] == "u_priya", acme
        bad = await client.call_tool("create_ticket", {"account_id": "acc_acme"})
        assert bad.isError, bad
        print("OK  tools:", names)
        print("OK  get_account(acc_acme) ->", acme.structuredContent)
        print("OK  bad call ->", bad.structuredContent)


anyio.run(main)
