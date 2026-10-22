"""Server tests: tool registration, tool-call behavior over a fake endpoint."""

import asyncio
import json
import unittest

from weigh_n_decide.config import Config
from weigh_n_decide.server import create_server
from tests.test_engines import ANSWERS, FakeServer, QUESTIONS


def cfg_for(server, **overrides):
    defaults = dict(
        engine="selfhosted",
        endpoint=server.url,
        model="clef",
        fallbacks=(),
        openrouter_url="http://openrouter.test/api/v1/systemone",
        openrouter_key="test-key",
        timeout=5.0,
    )
    defaults.update(overrides)
    return Config(**defaults)


def call(mcp, name, arguments):
    result = asyncio.run(mcp.call_tool(name, arguments))
    return json.loads(result.content[0].text)


def tool_names(mcp) -> set[str]:
    return {t.name for t in asyncio.run(mcp.list_tools())}


class RegistrationTests(unittest.TestCase):
    def test_registers_all_three_tools(self):
        mcp = create_server(cfg_for(FakeServer()))  # server never called
        self.assertEqual(tool_names(mcp), {"decide", "vision_decide", "evaluate"})


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

    def test_decide_without_endpoint_is_clear_error(self):
        mcp = create_server(Config(engine="selfhosted", endpoint=None))
        out = call(mcp, "decide", {"state": "msg", "questions": QUESTIONS})
        self.assertIn("error", out)
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", out["error"])

    def test_decide_bad_questions_is_error_json(self):
        mcp = create_server(cfg_for(self.server))
        out = call(mcp, "decide", {"state": "msg", "questions": {"x": {"type": "essay"}}})
        self.assertIn("error", out)

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


if __name__ == "__main__":
    unittest.main()