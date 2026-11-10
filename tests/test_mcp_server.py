"""The one MCP test promised in D1: the MCP path must be indistinguishable from the Python path."""
import anyio
from mcp.shared.memory import create_connected_server_and_client_session

from opsgym.mcp_server import build_server
from opsgym.seeds import SSO_SEED
from opsgym.tools import TOOLS, call_tool
from opsgym.workspace import load_seed

GOOD = {"account_id": "acc_acme", "title": "SSO down", "priority": "P1", "owner_id": "u_priya"}


def python_path(name, args):
    return call_tool(load_seed(SSO_SEED), name, args)


def test_mcp_path_matches_python_path():
    async def run():
        async with create_connected_server_and_client_session(build_server(SSO_SEED)) as client:
            published = {t.name: (t.description, t.inputSchema) for t in (await client.list_tools()).tools}
            assert published == {n: (t["description"], t["input_schema"]) for n, t in TOOLS.items()}

            for name, args in [("create_ticket", GOOD),                          # success
                               ("create_ticket", {**GOOD, "priority": "high"}),  # schema error
                               ("create_ticket", {**GOOD, "owner_id": "nobody"}),  # unknown ID
                               ("delete_everything", {})]:                       # hallucinated
                expected = python_path(name, args)
                got = await client.call_tool(name, args)
                assert got.structuredContent == expected, name
                assert got.isError == ("error" in expected), name
    anyio.run(run)
