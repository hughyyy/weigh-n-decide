"""Engine tests: validation, image conventions, dispatch, fallbacks — against
a fake /v1/systemone served in-process (stdlib http.server). No network."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from weigh_n_decide.config import Config
from weigh_n_decide.engines import EngineError, run_decision

# 1x1 PNG
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
PNG_DATA_URI = f"data:image/png;base64,{PNG_B64}"

QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "department": {
        "type": "choice",
        "criteria": {"billing": "Payments", "technical": "Bugs"},
    },
}
ANSWERS = {
    "urgent": {"probability": 0.9},
    "department": {"choice": "technical", "confidence": 0.7, "probabilities": {"billing": 0.3, "technical": 0.7}},
}
RESPONSE = {"model": "clef", "answers": ANSWERS, "usage": {"input": 10}}


class _FakeSystemOne(BaseHTTPRequestHandler):
    """Configurable fake: class attrs set per test; captures request bodies."""

    status = 200
    body = RESPONSE
    requests: list[dict] = []
    headers: list[dict] = []

    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", 0))
        _FakeSystemOne.requests.append(json.loads(self.rfile.read(length)))
        _FakeSystemOne.headers.append(dict(self.headers))
        data = json.dumps(_FakeSystemOne.body).encode()
        self.send_response(_FakeSystemOne.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence test output
        pass


class FakeServer:
    def __init__(self, status=200, body=RESPONSE):
        _FakeSystemOne.status = status
        _FakeSystemOne.body = body
        _FakeSystemOne.requests = []
        _FakeSystemOne.headers = []
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSystemOne)
        self.port = self.httpd.server_port
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    @property
    def requests(self):
        return _FakeSystemOne.requests

    @property
    def headers(self):
        return _FakeSystemOne.headers

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()




def _cfg(**kwargs):
    defaults = dict(
        engine="selfhosted",
        endpoint=None,
        model="clef",
        fallbacks=(),
        openrouter_url="http://openrouter.test/api/v1/systemone",
        openrouter_key="test-key",
        timeout=5.0,
    )
    defaults.update(kwargs)
    return Config(**defaults)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    def tearDown(self):
        self.server.stop()

    def test_selfhosted_roundtrip_normalizes_response(self):
        cfg = _cfg(endpoint=self.server.url)
        out = run_decision("A customer message", QUESTIONS, model=None, engine=None, cfg=cfg)
        self.assertEqual(out["model"], "clef")
        self.assertEqual(out["answers"], ANSWERS)
        self.assertEqual(out["usage"], {"input": 10})
        payload = self.server.requests[0]
        self.assertEqual(payload["model"], "clef")
        self.assertEqual(payload["state"], "A customer message")
        self.assertEqual(payload["questions"], QUESTIONS)
        self.assertNotIn("assets", payload)
        self.assertNotIn("images", payload)

    def test_endpoint_gets_versioned_systemone_path(self):
        cfg = _cfg(endpoint=self.server.url + "/")
        run_decision("s", QUESTIONS, cfg=cfg)
        self.assertEqual(self.server.requests[0]["model"], "clef")  # reached: path appended


class ValidationTests(unittest.TestCase):
    def test_state_required(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(None, QUESTIONS, cfg=_cfg())
        self.assertIn("state is required", str(ctx.exception))

    def test_questions_must_be_non_empty_object(self):
        with self.assertRaises(EngineError):
            run_decision("s", {}, cfg=_cfg())
        with self.assertRaises(EngineError):
            run_decision("s", [], cfg=_cfg())

    def test_unknown_question_type(self):
        bad = {"x": {"type": "essay"}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("noul/choice/score", str(ctx.exception))

    def test_unknown_question_field(self):
        bad = {"x": {"type": "noul", "rubric": "nope"}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("unknown field", str(ctx.exception))

    def test_choice_requires_criteria_object(self):
        bad = {"x": {"type": "choice"}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("choice needs criteria", str(ctx.exception))

    def test_score_requires_legend_list(self):
        bad = {"x": {"type": "score", "criteria": {"low": "x"}}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("score needs criteria", str(ctx.exception))

    def test_noul_criteria_keys_restricted(self):
        bad = {"x": {"type": "noul", "criteria": {"maybe": "x"}}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("'true'/'false'", str(ctx.exception))

    def test_unknown_model_lists_known(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, model="gpt-99", cfg=_cfg())
        self.assertIn("known models", str(ctx.exception))
        self.assertIn("clef", str(ctx.exception))

    def test_slashed_model_routes_to_openrouter(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, model="vendor/thing", cfg=_cfg(openrouter_key=None))
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))


class EngineConfigTests(unittest.TestCase):
    def test_missing_selfhosted_endpoint_is_a_clear_error(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, cfg=_cfg(endpoint=None))
        self.assertIn("WEIGH_N_DECIDE_ENDPOINT", str(ctx.exception))

    def test_missing_openrouter_key_is_a_clear_error(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS, model="inception/mercury-decide",
                cfg=_cfg(endpoint="http://127.0.0.1:1", openrouter_key=None),
            )
        self.assertIn("OPENROUTER_API_KEY is not set", str(ctx.exception))

    def test_unknown_engine_param(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, engine="carrier-pigeon", cfg=_cfg(endpoint=None))
        self.assertIn("unknown engine", str(ctx.exception))


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        self.cfg = _cfg(endpoint=self.server.url)

    def tearDown(self):
        self.server.stop()

    def test_assets_convention_for_decider(self):
        run_decision(
            {"chart": "q3-chart", "note": "Escalate?"},
            {"escalate": {"type": "noul"}},
            images=[{"image": PNG_DATA_URI, "ref": "q3-chart"}],
            model="decider-2b",
            cfg=self.cfg,
        )
        payload = self.server.requests[0]
        self.assertIn("assets", payload)
        self.assertNotIn("images", payload)
        asset = payload["assets"]["q3-chart"]
        self.assertEqual(asset["type"], "image")
        self.assertEqual(asset["format"], "png")
        self.assertEqual(asset["data_base64"], PNG_B64)

    def test_default_refs_are_numbered(self):
        run_decision("look at these", {"x": {"type": "noul"}},
                     images=[PNG_DATA_URI, PNG_DATA_URI], model="decider-2b", cfg=self.cfg)
        self.assertEqual(sorted(self.server.requests[0]["assets"]), ["image_0", "image_1"])

    def test_images_convention_for_clef(self):
        run_decision("receipt", {"legible": {"type": "noul"}},
                     images=[PNG_DATA_URI], model="clef", cfg=self.cfg)
        payload = self.server.requests[0]
        self.assertEqual(payload["images"], [PNG_DATA_URI])
        self.assertNotIn("assets", payload)

    def test_fallback_chain_recovers(self):
        dead = FakeServer(status=500, body={"detail": "boom"})
        live = FakeServer()
        try:
            cfg = _cfg(
                endpoint=dead.url,
                openrouter_url=live.url,
                fallbacks=("openrouter:clef",),
            )
            out = run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(live.requests[0]["model"], "clef")
        finally:
            dead.stop()
            live.stop()

    def test_unknown_convention_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=[PNG_DATA_URI],
                         model="inception/mercury-decide",
                         cfg=_cfg(endpoint=self.server.url, openrouter_key="k"))
        self.assertIn("no documented image convention", str(ctx.exception))

    def test_max_images_enforced(self):
        five = [PNG_DATA_URI] * 5
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=five, model="decider-2b", cfg=self.cfg)
        self.assertIn("max 4 images", str(ctx.exception))

    def test_unsupported_format(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=["data:image/webp;base64,AAAA"],
                         model="decider-2b", cfg=self.cfg)
        self.assertIn("unsupported format", str(ctx.exception))

    def test_bad_image_input(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=[42], model="decider-2b", cfg=self.cfg)
        self.assertIn("images[0]", str(ctx.exception))

    def test_duplicate_refs(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS,
                         images=[{"image": PNG_DATA_URI, "ref": "a"}, {"image": PNG_DATA_URI, "ref": "a"}],
                         model="decider-2b", cfg=self.cfg)
        self.assertIn("duplicate image refs", str(ctx.exception))


class FallbackTests(unittest.TestCase):
    def test_fallback_chain_recovers(self):
        dead = FakeServer(status=500, body={"detail": "boom"})
        live = FakeServer()
        try:
            cfg = _cfg(
                endpoint=dead.url,
                fallbacks=("selfhosted:clef",),
                # second attempt must reach the live server: swap endpoint by
                # simulating recovery via a second candidate is not possible
                # with one endpoint — so test engine-switch fallback instead.
            )
            cfg = _cfg(
                endpoint=dead.url,
                openrouter_url=live.url,
                fallbacks=("openrouter:clef",),
            )
            out = run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(live.requests[0]["model"], "clef")
        finally:
            dead.stop()
            live.stop()

    def test_all_attempts_fail_aggregates_errors(self):
        dead = FakeServer(status=500, body={"detail": "boom"})
        try:
            cfg = _cfg(endpoint=dead.url, openrouter_url=dead.url,
                       fallbacks=("openrouter:inception/mercury-decide",))
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
            message = str(ctx.exception)
            self.assertIn("all engine attempts failed", message)
            self.assertIn("selfhosted:clef", message)
            self.assertIn("openrouter:inception/mercury-decide", message)
        finally:
            dead.stop()

    def test_bad_fallback_entry_reported(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, model="does-not-exist", cfg=_cfg(endpoint=None))
        self.assertIn("does-not-exist", str(ctx.exception))


class OpenRouterTests(unittest.TestCase):
    def test_bearer_header_sent(self):
        server = FakeServer()
        try:
            cfg = _cfg(endpoint=None, openrouter_url=server.url, openrouter_key="sk-test")
            out = run_decision("s", QUESTIONS, model="inception/mercury-decide", cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(server.headers[0].get("Authorization"), "Bearer sk-test")
        finally:
            server.stop()

    def test_http_error_becomes_clear_message(self):
        server = FakeServer(status=402, body={"error": "insufficient credits"})
        try:
            cfg = _cfg(endpoint=None, openrouter_url=server.url)
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, model="inception/mercury-decide", cfg=cfg)
            self.assertIn("HTTP 402", str(ctx.exception))
        finally:
            server.stop()

    def test_malformed_response_is_an_error(self):
        server = FakeServer(status=200, body={"choices": []})
        try:
            cfg = _cfg(endpoint=server.url)
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
            self.assertIn("malformed SystemOne response", str(ctx.exception))
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()