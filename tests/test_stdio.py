"""End-to-end stdio tests: the real MCP server over JSON-RPC pipes.

Spawns `python -m weigh_n_decide.server`, drives the actual MCP stdio
protocol (newline-delimited JSON-RPC): initialize, tools/list, tools/call.
No network, no keys, no endpoint — the exact AGENTS.md startup contract.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class StdioServer:
    """One live stdio server; newline-delimited JSON-RPC over pipes."""

    def __init__(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("WEIGH_N_DECIDE_") and k != "OPENROUTER_API_KEY"
        }
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "weigh_n_decide.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=str(ROOT),
        )
        self.messages = queue.Queue()
        self.next_id = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self.messages.put(json.loads(line))
            except ValueError:
                pass  # non-JSON protocol noise — ignored
        self.messages.put(None)  # EOF sentinel

    def _send(self, msg: dict) -> None:
        self.proc.stdin.write((json.dumps(msg) + "\n").encode())
        self.proc.stdin.flush()

    def notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def request(self, method: str, params: dict | None = None) -> dict:
        self.next_id += 1
        msg = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        while True:
            resp = self.messages.get(timeout=20)
            if resp is None:
                raise AssertionError("server closed stdout before responding")
            if resp.get("id") == self.next_id:
                return resp
            # unrelated notification or stale response — keep looking

    def stop(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        for stream in (self.proc.stdout, self.proc.stderr):
            if stream:
                stream.close()


class StdioEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = StdioServer()
        cls.tools = None

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def init(self):
        resp = self.server.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "weigh-n-decide-tests", "version": "0.0.0"},
            },
        )
        self.assertNotIn("error", resp)
        self.server.notify("notifications/initialized")
        return resp["result"]

    def list_tools(self):
        resp = self.server.request("tools/list")
        self.assertNotIn("error", resp)
        return {tool["name"]: tool for tool in resp["result"]["tools"]}

    # -- protocol handshake ------------------------------------------------

    def test_initialize_handshake(self):
        result = self.init()
        self.assertEqual(result["serverInfo"]["name"], "weigh-n-decide")
        self.assertIn("tools", result["capabilities"])

    def test_tools_list_registers_all_three(self):
        self.init()
        tools = self.list_tools()
        self.assertEqual(set(tools), {"decide", "vision_decide", "evaluate"})
        # Tool schemas advertise required params.
        self.assertIn("state", tools["decide"]["inputSchema"]["properties"])
        self.assertIn("images", tools["vision_decide"]["inputSchema"]["properties"])
        self.assertIn("conditions", tools["evaluate"]["inputSchema"]["properties"])

    # -- tools/call behavior ----------------------------------------------

    def call(self, name, arguments):
        resp = self.server.request("tools/call", {"name": name, "arguments": arguments})
        self.assertNotIn("error", resp)
        text = resp["result"]["content"][0]["text"]
        return json.loads(text)

    def test_evaluate_works_over_stdlib(self):
        self.init()
        out = self.call(
            "evaluate",
            {"conditions": ["a > 1", "b == 'x'"], "facts": {"a": 5, "b": "x"}},
        )
        self.assertEqual(
            [c["result"] for c in out["conditions"]], [True, True]
        )
        self.assertIs(out["all"], True)
        self.assertIs(out["any"], True)

    def test_evaluate_missing_fact_is_a_structured_error(self):
        self.init()
        out = self.call("evaluate", {"conditions": "missing.path > 1", "facts": {}})
        self.assertIn("error", out)
        self.assertIn("missing key 'missing'", out["error"])

    def test_decide_without_endpoint_errors_clearly(self):
        self.init()
        out = self.call(
            "decide",
            {"state": "s", "questions": {"x": {"type": "noul"}}},
        )
        self.assertIn("error", out)
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", out["error"])

    def test_decide_accepts_a_numeric_state(self):
        # The tool schema mirrors the questions API: state is a string or any
        # JSON value, so the call must reach the engine (and fail on the
        # missing endpoint), not on argument validation.
        self.init()
        out = self.call(
            "decide",
            {"state": 42, "questions": {"x": {"type": "noul"}}},
        )
        self.assertIn("error", out)
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", out["error"])

    def test_vision_decide_without_endpoint_errors_clearly(self):
        self.init()
        out = self.call(
            "vision_decide",
            {"images": ["data:image/png;base64,AAAA"], "state": "s",
             "questions": {"x": {"type": "noul"}}, "model": "decider-2b"},
        )
        self.assertIn("error", out)
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", out["error"])

    def test_unknown_tool_is_a_tool_error(self):
        self.init()
        resp = self.server.request("tools/call", {"name": "nope", "arguments": {}})
        self.assertIs(resp["result"]["isError"], True)
        self.assertIn("Unknown tool: nope", resp["result"]["content"][0]["text"])

    def test_ping_roundtrip(self):
        self.init()
        resp = self.server.request("ping")
        self.assertNotIn("error", resp)
        self.assertEqual(resp["result"], {})

    def test_missing_required_argument_is_a_tool_error(self):
        self.init()
        resp = self.server.request("tools/call", {"name": "decide", "arguments": {}})
        self.assertIs(resp["result"]["isError"], True)
        self.assertIn("state", resp["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
