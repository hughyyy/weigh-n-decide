"""Server tests: tool registration, tool-call behavior over a fake endpoint."""

import asyncio
import json
import unittest
from pathlib import Path

from weigh_n_decide.config import Config
from weigh_n_decide.server import create_server
from tests.test_engines import ANSWERS, FakeServer, QUESTIONS


def cfg_for(server, **overrides):
    defaults = dict(
        engine=None,
        endpoint=server.url,
        model="clef",
        fallbacks=(),
        openrouter_url="http://openrouter.test/api/v1/systemone",
        openrouter_key="test-key",
        timeout=5.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def registration_cfg() -> Config:
    return Config(
        engine=None,
        endpoint="http://127.0.0.1:1",
        model="clef",
        openrouter_url="http://127.0.0.1:2/api/v1/systemone",
        openrouter_key="test-key",
        timeout=5.0,
    )


def call(mcp, name, arguments):
    result = asyncio.run(mcp.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def tool_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


class RegistrationTests(unittest.TestCase):
    def test_registers_all_three_tools(self):
        mcp = create_server(registration_cfg())
        self.assertEqual(
            tool_names(mcp), {"decide", "vision_decide", "evaluate"}
        )

    def test_tool_schemas_advertise_required_parameters(self):
        mcp = create_server(registration_cfg())
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        self.assertEqual(
            set(tools["decide"].input_schema["required"]),
            {"state", "questions"},
        )
        self.assertEqual(
            set(tools["vision_decide"].input_schema["required"]),
            {"images", "state", "questions"},
        )
        self.assertEqual(
            set(tools["evaluate"].input_schema["required"]),
            {"conditions"},
        )

    def test_state_schema_documents_every_json_scalar(self):
        mcp = create_server(registration_cfg())
        tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
        for name in ("decide", "vision_decide"):
            state_schema = tools[name].input_schema["properties"]["state"]
            self.assertEqual(
                {entry["type"] for entry in state_schema["anyOf"]},
                {"boolean", "integer", "number", "string", "array", "object"},
            )


class EvaluateToolTests(unittest.TestCase):
    def test_evaluate_returns_structured_json(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": "a > 5", "facts": {"a": 7}})
        self.assertIs(out["conditions"][0]["result"], True)
        self.assertIs(out["all"], True)

    def test_evaluate_condition_error_is_json(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": "missing.thing > 1", "facts": {}})
        self.assertIn("error", out)
        self.assertIn("missing key 'missing'", out["error"])

    def test_evaluate_list_and_aggregates(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": ["a > 1", "b > 1"], "facts": {"a": 2, "b": 0}})
        self.assertIs(out["all"], False)
        self.assertIs(out["any"], True)

    def test_evaluate_never_touches_the_network(self):
        from unittest import mock

        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        with mock.patch("httpx.post", side_effect=AssertionError("network call")):
            out = call(mcp, "evaluate", {"conditions": "a > 1", "facts": {"a": 2}})
        self.assertIs(out["conditions"][0]["result"], True)

    def test_evaluate_is_repeatable(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        args = {"conditions": ["a > 1", "b contains 'x'"], "facts": {"a": 2, "b": ["x"]}}
        self.assertEqual(call(mcp, "evaluate", args), call(mcp, "evaluate", args))

    def test_evaluate_empty_condition_list_is_error_json(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": []})
        self.assertIn("error", out)
        self.assertIn("non-empty", out["error"])


class DecideToolTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    def tearDown(self):
        self.server.stop()

    def test_decide_roundtrip(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": "msg", "questions": QUESTIONS})
        self.assertEqual(out["answers"], ANSWERS)
        self.assertEqual(out["model"], "clef")

    def test_decide_accepts_every_documented_state_shape(self):
        mcp = create_server(cfg_for(self.server))
        for state in (5, True, 3.5, "text", [1, "x"], {"a": 1}):
            with self.subTest(state=state):
                out = call(mcp, "decide", {"state": state, "questions": QUESTIONS})
                self.assertEqual(out["answers"], ANSWERS)
                self.assertEqual(self.server.requests[-1]["state"], state)

    def test_vision_decide_accepts_numeric_state(self):
        png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        mcp = create_server(cfg_for(self.server, model="decider-2b"))
        out = call(
            mcp,
            "vision_decide",
            {"images": [png], "state": 7, "questions": {"x": {"type": "noul"}}},
        )
        self.assertEqual(out["answers"], ANSWERS)
        self.assertEqual(self.server.requests[0]["state"], 7)

    def test_decide_unserializable_state_is_error_json(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": {"x": object()}, "questions": QUESTIONS})
        self.assertIn("error", out)
        self.assertIn("not JSON serializable", out["error"])

    def test_decide_without_endpoint_is_clear_error(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "decide", {"state": "msg", "questions": QUESTIONS})
        self.assertIn("error", out)
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", out["error"])

    def test_decide_bad_questions_is_error_json(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": "msg", "questions": {"x": {"type": "essay"}}})
        self.assertIn("error", out)

    def test_decide_unexpected_failure_becomes_error_json(self):
        from unittest import mock

        import weigh_n_decide.server as server_module

        mcp = create_server(cfg_for(self.server))
        with mock.patch.object(
            server_module.engines, "run_decision", side_effect=RuntimeError("boom")
        ):
            out = call(mcp, "decide", {"state": "s", "questions": QUESTIONS})
        self.assertEqual(out, {"error": "boom"})

    def test_vision_decide_routes_images(self):
        mcp = create_server(cfg_for(self.server, model="decider-2b"))
        png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        out = call(
            mcp,
            "vision_decide",
            {"images": [png], "state": "receipt", "questions": {"legible": {"type": "noul"}}},
        )
        self.assertEqual(out["answers"], ANSWERS)
        self.assertIn("assets", self.server.requests[0])

    def test_vision_decide_non_vision_model_is_error(self):
        mcp = create_server(cfg_for(self.server, model="decider-12b"))
        png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        out = call(
            mcp,
            "vision_decide",
            {"images": [png], "state": "s", "questions": QUESTIONS},
        )
        self.assertIn("error", out)
        self.assertIn("no image support", out["error"])

    def test_vision_decide_clef_uses_images_convention(self):
        mcp = create_server(cfg_for(self.server, model="clef"))
        png = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        out = call(
            mcp,
            "vision_decide",
            {"images": [png], "state": "s", "questions": {"legible": {"type": "noul"}}},
        )
        self.assertEqual(out["answers"], ANSWERS)
        payload = self.server.requests[0]
        self.assertEqual(payload["images"], [png])
        self.assertNotIn("assets", payload)

    def test_decide_empty_questions_is_error_json(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": "s", "questions": {}})
        self.assertIn("error", out)
        self.assertIn("non-empty object", out["error"])

    def test_evaluate_without_facts_uses_empty_objects(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": "true"})
        self.assertIs(out["conditions"][0]["result"], True)
        self.assertIs(out["all"], True)

    def test_evaluate_bad_conditions_type_is_error_json(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        # `5` is rejected by the tool schema before evaluation; schema-valid
        # but semantically bad inputs become structured error JSON.
        out = call(mcp, "evaluate", {"conditions": ""})
        self.assertIn("error", out)
        self.assertIn("non-empty", out["error"])
        out = call(mcp, "evaluate", {"conditions": [5]})
        self.assertIn("error", out)
        self.assertIn("each condition must be a string", out["error"])

    def test_vision_decide_missing_images_is_a_tool_error(self):
        from mcp.server.mcpserver.exceptions import ToolError

        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        with self.assertRaises(ToolError):
            asyncio.run(
                mcp.call_tool(
                    "vision_decide", {"state": "s", "questions": QUESTIONS}
                )
            )

    def test_evaluate_missing_conditions_is_a_tool_error(self):
        from mcp.server.mcpserver.exceptions import ToolError

        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        with self.assertRaises(ToolError):
            asyncio.run(mcp.call_tool("evaluate", {}))

    def test_evaluate_rejects_non_object_facts(self):
        from mcp.server.mcpserver.exceptions import ToolError

        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        with self.assertRaises(ToolError):
            asyncio.run(
                mcp.call_tool("evaluate", {"conditions": "true", "facts": [1]})
            )

    def test_unserializable_fact_becomes_error_json(self):
        # A non-JSON value in facts must not produce a traceback: the
        # structured error is the tool contract.
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "evaluate", {"conditions": "x == 1", "facts": {"x": object()}})
        self.assertIn("error", out)
        self.assertIn("not JSON serializable", out["error"])

    def test_tool_missing_required_argument_raises(self):
        from mcp.server.mcpserver.exceptions import ToolError

        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        with self.assertRaises(ToolError):
            asyncio.run(mcp.call_tool("decide", {}))

    def test_decide_model_param_overrides_config(self):
        mcp = create_server(cfg_for(self.server, model="clef"))
        out = call(mcp, "decide", {"state": "s", "questions": QUESTIONS,
                                   "model": "decider-31b"})
        self.assertEqual(out["answers"], ANSWERS)
        self.assertEqual(self.server.requests[0]["model"], "decider-31b")

    def test_decide_unknown_engine_param_is_error_json(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": "s", "questions": QUESTIONS,
                                   "engine": "carrier-pigeon"})
        self.assertIn("error", out)
        self.assertIn("unknown engine", out["error"])


class MainEntryTests(unittest.TestCase):
    def test_main_runs_the_stdio_transport(self):
        from unittest import mock

        import weigh_n_decide.server as server_module

        # Patch create_server away so main() cannot block on a real stdio loop.
        calls = {}
        fake = mock.Mock()
        fake.run.side_effect = lambda **kw: calls.update(kw)
        with mock.patch.object(server_module, "create_server", return_value=fake):
            server_module.main()
        fake.run.assert_called_once_with(transport="stdio")

    def test_package_entry_calls_main(self):
        # `python -m weigh_n_decide` routes through __main__.py; patch main
        # so the import-time call is observable without starting stdio.
        import runpy
        from unittest import mock

        import weigh_n_decide.server as server_module

        with mock.patch.object(server_module, "main") as main:
            runpy.run_module("weigh_n_decide", run_name="__main__")
        main.assert_called_once_with()

    def test_module_entry_starts_and_exits_on_stdin_eof(self):
        # `python -m weigh_n_decide.server` bootstraps with no keys and no
        # endpoint (registration is startup-safe), then exits when stdin
        # closes. Guards the AGENTS.md no-keys startup contract.
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "weigh_n_decide.server"],
            input=b"",
            timeout=60,
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertEqual(proc.stdout, b"")

    def test_package_module_entry_starts_and_exits_on_stdin_eof(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "weigh_n_decide"],
            input=b"",
            timeout=60,
            capture_output=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())
        self.assertEqual(proc.stdout, b"")


if __name__ == "__main__":
    unittest.main()