"""Manual smoke: drive weigh-n-decide over real MCP stdio.

Usage: .venv/bin/python scripts/stdio_smoke.py
Verifies: initialize, tools/list, evaluate roundtrip, clear error paths.
"""

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run() -> int:
    params = StdioServerParameters(command=sys.executable, args=["-m", "weigh_n_decide"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", sorted(t.name for t in tools.tools))

            result = await session.call_tool(
                "evaluate",
                {
                    "conditions": [
                        'a.status == "paid" and (a.total > 100 or a.trial) and not a.vip',
                        "items[0].price < 5",
                        "'refund' in tags",
                    ],
                    "facts": {
                        "a": {"status": "paid", "total": 250, "vip": False},
                        "items": [{"price": 4.5}],
                        "tags": ["refund", "vip"],
                    },
                },
            )
            print("evaluate:", result.content[0].text)

            result = await session.call_tool(
                "decide",
                {"state": "msg", "questions": {"x": {"type": "essay"}}},
            )
            print("decide-bad-type:", result.content[0].text)

            result = await session.call_tool(
                "decide",
                {"state": "msg", "questions": {"urgent": {"type": "noul"}}},
            )
            print("decide-no-endpoint:", result.content[0].text)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run()))