"""Engine tests: validation, image conventions, dispatch, fallbacks — against
a fake /v1/systemone served in-process (stdlib http.server). No network."""

import json
import os
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
    """Configurable fake: state is keyed by server address so multiple
    FakeServers (fallback chains) keep independent status/body/logs."""

    states: dict = {}
    requests: dict = {}
    paths: dict = {}
    headers: dict = {}

    def _key(self):
        return self.server.server_address[:2]

    def do_POST(self):  # noqa: N802 - http.server API
        key = self._key()
        length = int(self.headers.get("Content-Length", 0))
        _FakeSystemOne.requests[key].append(json.loads(self.rfile.read(length)))
        _FakeSystemOne.paths[key].append(self.path)
        _FakeSystemOne.headers[key].append(dict(self.headers))
        state = _FakeSystemOne.states[key]
        raw = state["raw_body"]
        data = raw if raw is not None else json.dumps(state["body"]).encode()
        self.send_response(state["status"])
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence test output
        pass


class FakeServer:
    def __init__(self, status=200, body=RESPONSE, raw_body=None):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _FakeSystemOne)
        key = self.httpd.server_address[:2]
        _FakeSystemOne.states[key] = {
            "status": status,
            "body": body,
            "raw_body": raw_body,
        }
        _FakeSystemOne.requests[key] = []
        _FakeSystemOne.paths[key] = []
        _FakeSystemOne.headers[key] = []
        self.thread = threading.Thread(
            target=lambda: self.httpd.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.httpd.server_port}"

    @property
    def key(self):
        return self.httpd.server_address[:2]

    @property
    def requests(self):
        return _FakeSystemOne.requests[self.key]

    @property
    def paths(self):
        return _FakeSystemOne.paths[self.key]

    @property
    def headers(self):
        return _FakeSystemOne.headers[self.key]

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        _FakeSystemOne.states.pop(self.key, None)
        _FakeSystemOne.requests.pop(self.key, None)
        _FakeSystemOne.paths.pop(self.key, None)
        _FakeSystemOne.headers.pop(self.key, None)




def _cfg(**kwargs):
    defaults = dict(
        engine=None,
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
        self.assertEqual(self.server.paths[0], "/v1/systemone")


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

    def test_blank_model_names_are_rejected(self):
        for bad in (None, "", "   "):
            with self.assertRaises(EngineError, msg=repr(bad)) as ctx:
                run_decision("s", QUESTIONS, model=bad, cfg=_cfg(model="  "))
            self.assertIn("model name is empty", str(ctx.exception))

    def test_slashed_model_routes_to_openrouter(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, model="vendor/thing", cfg=_cfg(openrouter_key=None))
        self.assertIn("OPENROUTER_API_KEY", str(ctx.exception))

    def test_choice_criteria_must_not_be_empty(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", {"x": {"type": "choice", "criteria": {}}},
                         cfg=_cfg(endpoint="http://127.0.0.1:1"))
        self.assertIn("choice needs criteria", str(ctx.exception))

    def test_score_criteria_must_not_be_empty(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", {"x": {"type": "score", "criteria": []}},
                         cfg=_cfg(endpoint="http://127.0.0.1:1"))
        self.assertIn("score needs criteria", str(ctx.exception))


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

    def test_mixed_image_entries_keep_order_and_refs(self):
        run_decision(
            "s", {"x": {"type": "noul"}},
            images=[PNG_DATA_URI, {"image": PNG_DATA_URI, "ref": "chart"}],
            model="decider-2b", cfg=self.cfg,
        )
        self.assertEqual(list(self.server.requests[0]["assets"]), ["image_0", "chart"])

    def test_explicit_ref_colliding_with_a_default_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS,
                images=[PNG_DATA_URI, {"image": PNG_DATA_URI, "ref": "image_0"}],
                model="decider-2b", cfg=self.cfg,
            )
        self.assertIn("duplicate image refs", str(ctx.exception))


class FallbackTests(unittest.TestCase):
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


class ValidationEdgeTests(unittest.TestCase):
    """Remaining validation branches: state types, question shapes, images."""

    def setUp(self):
        self.server = FakeServer()

    def tearDown(self):
        self.server.stop()

    def test_state_accepts_every_json_shape(self):
        cfg = _cfg(endpoint=self.server.url)
        for state in (True, 0, 3.5, "s", [], {}, {"a": 1}, [1, "x"]):
            out = run_decision(state, QUESTIONS, cfg=cfg)
            self.assertEqual(out["model"], "clef")  # reached the fake
            self.assertEqual(self.server.requests[-1]["state"], state)

    def test_state_non_json_type_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(object(), QUESTIONS, cfg=_cfg())
        self.assertIn("state must be a string or JSON value", str(ctx.exception))
        with self.assertRaises(EngineError) as ctx:
            run_decision(None, QUESTIONS, cfg=_cfg())
        self.assertIn("state is required", str(ctx.exception))

    def test_question_id_must_be_a_non_empty_string(self):
        for bad in ({"": {"type": "noul"}}, {1: {"type": "noul"}}, {None: {"type": "noul"}}):
            with self.assertRaises(EngineError, msg=repr(bad)) as ctx:
                run_decision("s", bad, cfg=_cfg())
            self.assertIn("question ids must be non-empty strings", str(ctx.exception))

    def test_question_must_be_an_object(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", {"x": "is it?"}, cfg=_cfg())
        self.assertIn("must be an object", str(ctx.exception))

    def test_instructions_must_be_a_string(self):
        bad = {"x": {"type": "noul", "instructions": 42}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("instructions must be a string", str(ctx.exception))

    def test_choice_option_description_must_be_a_string(self):
        bad = {"x": {"type": "choice", "criteria": {"a": "ok", "b": 5}}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("description must be a string", str(ctx.exception))

    def test_score_legend_entries_must_be_strings(self):
        bad = {"x": {"type": "score", "criteria": ["low", 5]}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("legend entries must be strings", str(ctx.exception))

    def test_noul_criteria_must_be_an_object(self):
        bad = {"x": {"type": "noul", "criteria": "probably"}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("noul criteria must be an object", str(ctx.exception))

    def test_noul_criteria_descriptions_must_be_strings(self):
        bad = {"x": {"type": "noul", "criteria": {"true": 1}}}
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", bad, cfg=_cfg())
        self.assertIn("criteria 'true' must be a string", str(ctx.exception))

    def test_images_must_be_a_non_empty_list(self):
        for bad in ({}, [], "one.png", 5):
            with self.assertRaises(EngineError, msg=repr(bad)) as ctx:
                run_decision("s", QUESTIONS, images=bad, cfg=self.cfg_for_vision())
            self.assertIn("images must be a non-empty list", str(ctx.exception))

    def test_image_entry_needs_an_image_field(self):
        for bad in ({"ref": "a"}, {"image": ""}):
            with self.assertRaises(EngineError, msg=repr(bad)) as ctx:
                run_decision("s", QUESTIONS, images=[bad], cfg=self.cfg_for_vision())
            self.assertIn("'image' must be a non-empty string", str(ctx.exception))
        # Whitespace-only survives the engine check and is rejected by
        # load_image with its own clear message.
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=[{"image": "   "}],
                         cfg=self.cfg_for_vision())
        self.assertIn("non-empty string", str(ctx.exception))

    def test_image_ref_must_be_a_non_empty_string(self):
        for bad in ({"image": PNG_DATA_URI, "ref": ""}, {"image": PNG_DATA_URI, "ref": 5}):
            with self.assertRaises(EngineError, msg=repr(bad)) as ctx:
                run_decision("s", QUESTIONS, images=[bad], cfg=self.cfg_for_vision())
            self.assertIn("'ref' must be a non-empty string", str(ctx.exception))

    def test_unreadable_image_file_is_wrapped(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, images=["/no/such/image.png"],
                         cfg=self.cfg_for_vision())
        self.assertIn("images[0] (image_0)", str(ctx.exception))
        self.assertIn("cannot read image file", str(ctx.exception))

    def test_build_payload_rejects_unknown_convention(self):
        # run_decision pre-filters convention=None; a convention the registry
        # never emits is still a clear error, not a silent omission.
        from types import SimpleNamespace

        from weigh_n_decide.engines import _build_payload
        from weigh_n_decide.registry import ModelInfo

        img = SimpleNamespace(ref="a", media="image/png", b64="AA", fmt="png")
        info = ModelInfo(engine="selfhosted", vision=True, convention="bogus")
        with self.assertRaises(EngineError) as ctx:
            _build_payload("clef", "s", QUESTIONS, [img], info)
        self.assertIn("no documented image convention", str(ctx.exception))

    def test_valid_score_and_noul_criteria_pass_through(self):
        # Clean validation-loop iterations: score with a valid legend and
        # noul with valid true/false criteria never trip an error.
        questions = {
            "rate": {
                "type": "score",
                "instructions": "rate it",
                "criteria": ["low", "mid", "high"],
            },
            "confirm": {"type": "noul",
                        "criteria": {"true": "is so", "false": "is not"}},
        }
        out = run_decision("s", questions, cfg=_cfg(endpoint=self.server.url))
        self.assertEqual(out["model"], "clef")
        request = self.server.requests[0]
        self.assertEqual(request["questions"]["rate"]["criteria"], ["low", "mid", "high"])
        self.assertEqual(request["questions"]["confirm"]["criteria"], {"true": "is so", "false": "is not"})

    def test_score_question_full_roundtrip(self):
        questions = {"rate": {"type": "score", "criteria": ["bad", "ok", "great"]}}
        answers = {"rate": {"score": 2, "confidence": 0.8,
                            "legend": ["bad", "ok", "great"],
                            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}
        server = FakeServer(body={"model": "clef", "answers": answers, "usage": None})
        try:
            out = run_decision("s", questions, cfg=_cfg(endpoint=server.url))
            self.assertEqual(out["answers"], answers)
            self.assertIsNone(out["usage"])
        finally:
            server.stop()

    def cfg_for_vision(self):
        return _cfg(endpoint=self.server.url, model="decider-2b")


class DispatchEdgeTests(unittest.TestCase):
    def test_connection_refused_is_a_clear_error(self):
        import socket

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        cfg = _cfg(endpoint=f"http://127.0.0.1:{port}")
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, cfg=cfg)
        self.assertIn("request to http://127.0.0.1:", str(ctx.exception))
        self.assertIn("failed", str(ctx.exception))

    def test_non_json_response_is_a_clear_error(self):
        server = FakeServer(raw_body=b"{this is not json")
        try:
            cfg = _cfg(endpoint=server.url)
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
            self.assertIn("response from", str(ctx.exception))
            self.assertIn("was not JSON", str(ctx.exception))
        finally:
            server.stop()

    def test_bare_model_fallback_uses_models_own_engine(self):
        dead = FakeServer(status=500, body={"detail": "boom"})
        live = FakeServer()
        try:
            cfg = _cfg(
                endpoint=dead.url,
                openrouter_url=live.url,
                fallbacks=("inception/mercury-decide",),
            )
            out = run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            # FakeServers keep independent logs; live.requests[0] is the
            # recovered fallback attempt.
            self.assertEqual(live.requests[0]["model"], "inception/mercury-decide")
            self.assertIn("Authorization", live.headers[0])
        finally:
            dead.stop()
            live.stop()

    def test_fallback_to_unknown_model_reports_it(self):
        cfg = _cfg(endpoint=None, fallbacks=("no-such-model",))
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, cfg=cfg)
        self.assertIn("no-such-model", str(ctx.exception))

    def test_duplicate_candidates_are_deduped(self):
        # Same (engine, model) from default and fallback must produce one
        # HTTP attempt, not two.
        server = FakeServer()
        try:
            cfg = _cfg(endpoint=server.url, fallbacks=("selfhosted:clef",))
            run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(len(server.requests), 1)
        finally:
            server.stop()

    def test_unknown_response_fields_are_dropped(self):
        server = FakeServer(body={"model": "clef", "answers": ANSWERS,
                                  "usage": {"input": 1}, "extra": "junk"})
        try:
            out = run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
        finally:
            server.stop()
        self.assertEqual(set(out), {"model", "answers", "usage"})

    def test_answers_not_an_object_is_malformed(self):
        server = FakeServer(body={"answers": [{"probability": 0.5}]})
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
            self.assertIn("malformed SystemOne response", str(ctx.exception))
        finally:
            server.stop()

    def test_non_200_with_empty_body_is_a_clear_error(self):
        server = FakeServer(status=204, body=None)
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
            self.assertIn("HTTP 204", str(ctx.exception))
        finally:
            server.stop()

    def test_multiple_images_reach_clef_convention(self):
        server = FakeServer()
        try:
            run_decision("s", QUESTIONS,
                         images=[PNG_DATA_URI, PNG_DATA_URI],
                         model="clef", cfg=_cfg(endpoint=server.url))
            payload = server.requests[0]
            self.assertEqual(payload["images"], [PNG_DATA_URI, PNG_DATA_URI])
        finally:
            server.stop()

    def test_response_defaults_model_and_usage(self):
        # A response without 'model'/'usage' falls back to the requested
        # model and None usage.
        server = FakeServer(body={"answers": ANSWERS})
        try:
            out = run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
            self.assertEqual(out["model"], "clef")
            self.assertIsNone(out["usage"])
        finally:
            server.stop()

    def test_engine_param_overrides_model_default_engine(self):
        # clef is a selfhosted model, but engine='openrouter' must win.
        server = FakeServer()
        try:
            cfg = _cfg(endpoint="http://127.0.0.1:1", openrouter_url=server.url)
            out = run_decision("s", QUESTIONS, model="clef", engine="openrouter", cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(server.requests[0]["model"], "clef")
            self.assertIn("Authorization", server.headers[0])
        finally:
            server.stop()

    def test_env_defaults_apply_at_server_composition_root(self):
        # create_server loads env once; run_decision also loads env when cfg
        # is omitted, so both composition paths honor WEIGH_N_DECIDE_*.
        from unittest.mock import patch

        server = FakeServer()
        try:
            with patch.dict(
                os.environ,
                {"WEIGH_N_DECIDE_ENDPOINT": server.url, "WEIGH_N_DECIDE_MODEL": "decider-2b"},
                clear=True,
            ):
                from weigh_n_decide.server import create_server

                mcp = create_server()  # cfg=None -> load_config()
            import asyncio

            result = asyncio.run(mcp.call_tool("decide",
                                               {"state": "s", "questions": QUESTIONS}))
            out = json.loads(result.content[0].text)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(server.requests[0]["model"], "decider-2b")
        finally:
            server.stop()


class ImageConventionEdgeTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()

    def tearDown(self):
        self.server.stop()

    def test_jpg_media_maps_to_jpeg_asset_format(self):
        uri = f"data:image/jpg;base64,{PNG_B64}"
        run_decision(
            {"chart": "c"}, {"escalate": {"type": "noul"}},
            images=[{"image": uri, "ref": "c"}], model="decider-2b",
            cfg=_cfg(endpoint=self.server.url),
        )
        asset = self.server.requests[0]["assets"]["c"]
        self.assertEqual(asset["format"], "jpeg")
        self.assertEqual(asset["data_base64"], PNG_B64)

    def test_uppercase_media_type_maps_to_png_format(self):
        uri = f"data:image/PNG;base64,{PNG_B64}"
        run_decision(
            "s", {"x": {"type": "noul"}}, images=[uri], model="decider-2b",
            cfg=_cfg(endpoint=self.server.url),
        )
        self.assertEqual(self.server.requests[0]["assets"]["image_0"]["format"], "png")

    def test_clef_images_list_keeps_the_original_data_uri(self):
        uri = f"data:image/jpeg;base64,{PNG_B64}"
        run_decision(
            "s", {"x": {"type": "noul"}}, images=[uri], model="clef",
            cfg=_cfg(endpoint=self.server.url),
        )
        self.assertEqual(self.server.requests[0]["images"], [uri])

    def test_primary_success_never_reaches_the_fallback(self):
        fallback = FakeServer()
        try:
            cfg = _cfg(
                endpoint=self.server.url,
                openrouter_url=fallback.url,
                fallbacks=("openrouter:clef",),
            )
            run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(len(self.server.requests), 1)
            self.assertEqual(fallback.requests, [])
        finally:
            fallback.stop()

    def test_non_vision_candidate_is_skipped_for_a_vision_fallback(self):
        run_decision(
            "s", QUESTIONS, images=[PNG_DATA_URI], model="decider-12b",
            cfg=_cfg(endpoint=self.server.url, fallbacks=("selfhosted:clef",)),
        )
        self.assertEqual(len(self.server.requests), 1)
        self.assertIn("images", self.server.requests[0])

    def test_non_vision_candidate_error_names_the_limitation(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS, images=[PNG_DATA_URI], model="decider-12b",
                cfg=_cfg(endpoint=self.server.url),
            )
        self.assertIn("selfhosted:decider-12b: no image support", str(ctx.exception))

    def test_undocumented_convention_is_named_in_the_aggregate(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS, images=[PNG_DATA_URI],
                model="inception/mercury-decide",
                cfg=_cfg(openrouter_url=self.server.url),
            )
        message = str(ctx.exception)
        self.assertIn("no documented image convention", message)
        self.assertIn("clef", message)


class FallbackParsingEdgeTests(unittest.TestCase):
    def test_engine_only_fallback_uses_the_configured_model(self):
        server = FakeServer()
        try:
            cfg = _cfg(
                endpoint=None, openrouter_url=server.url, model="clef",
                fallbacks=("openrouter:",),
            )
            out = run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(server.requests[0]["model"], "clef")
        finally:
            server.stop()

    def test_fallback_without_engine_resolves_the_models_engine(self):
        server = FakeServer()
        try:
            cfg = _cfg(
                endpoint=None, openrouter_url=server.url,
                fallbacks=("inception/mercury-decide",),
            )
            run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(server.requests[0]["model"], "inception/mercury-decide")
            self.assertIn("Authorization", server.headers[0])
        finally:
            server.stop()

    def test_blank_engine_fallback_collapses_onto_the_primary(self):
        # ":clef" resolves to the model's own engine, so it dedupes with the
        # primary attempt instead of firing twice.
        server = FakeServer()
        try:
            run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url,
                                                  fallbacks=(" :clef ",)))
            self.assertEqual(len(server.requests), 1)
        finally:
            server.stop()

    def test_unknown_engine_in_fallback_is_aggregated(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS,
                cfg=_cfg(endpoint=None, fallbacks=("carrier-pigeon:clef",)),
            )
        message = str(ctx.exception)
        self.assertIn("all engine attempts failed", message)
        self.assertIn("unknown engine 'carrier-pigeon'", message)

    def test_fallback_model_with_colon_suffix_keeps_the_suffix(self):
        server = FakeServer()
        try:
            cfg = _cfg(
                endpoint=None, openrouter_url=server.url,
                fallbacks=("openrouter:inception/mercury-decide:free",),
            )
            run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(
                server.requests[0]["model"], "inception/mercury-decide:free"
            )
        finally:
            server.stop()


class DispatchContractEdgeTests(unittest.TestCase):
    def test_invalid_timeout_from_a_hand_built_config_is_an_engine_error(self):
        cfg = _cfg(endpoint="http://127.0.0.1:1", timeout=-1.0)
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, cfg=cfg)
        self.assertIn("request to http://127.0.0.1:1/v1/systemone failed",
                      str(ctx.exception))

    def test_endpoint_trailing_slashes_are_collapsed(self):
        server = FakeServer()
        try:
            run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url + "///"))
            self.assertEqual(server.requests[0]["model"], "clef")
        finally:
            server.stop()

    def test_exactly_max_images_is_accepted(self):
        server = FakeServer()
        try:
            run_decision(
                "s", QUESTIONS, images=[PNG_DATA_URI] * 4, model="decider-2b",
                cfg=_cfg(endpoint=server.url),
            )
            self.assertEqual(len(server.requests[0]["assets"]), 4)
        finally:
            server.stop()

    def test_read_timeout_is_an_engine_error(self):
        import httpx
        from unittest import mock

        cfg = _cfg(endpoint="http://127.0.0.1:1")
        with mock.patch(
            "weigh_n_decide.engines._CLIENT.post",
            side_effect=httpx.ReadTimeout("timed out"),
        ):
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
        self.assertIn("request to", str(ctx.exception))
        self.assertIn("timed out", str(ctx.exception))

    def test_unserializable_state_is_an_engine_error(self):
        cfg = _cfg(endpoint="http://127.0.0.1:1")
        with self.assertRaises(EngineError) as ctx:
            run_decision({"x": object()}, QUESTIONS, cfg=cfg)
        self.assertIn("not JSON serializable", str(ctx.exception))

    def test_redirect_status_is_a_clear_error(self):
        server = FakeServer(status=302, body=None)
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
            self.assertIn("HTTP 302", str(ctx.exception))
        finally:
            server.stop()

    def test_empty_openrouter_url_is_a_clear_error(self):
        cfg = _cfg(endpoint=None, openrouter_url="", openrouter_key="k")
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, model="inception/mercury-decide", cfg=cfg)
        self.assertIn("WEIGH_N_DECIDE_OPENROUTER_URL is empty", str(ctx.exception))

    def test_whitespace_only_fallback_is_deduped(self):
        server = FakeServer()
        try:
            run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url,
                                                  fallbacks=("   ",)))
            self.assertEqual(len(server.requests), 1)
        finally:
            server.stop()

    def test_empty_model_uses_the_configured_default(self):
        server = FakeServer()
        try:
            run_decision("s", QUESTIONS, model="",
                         cfg=_cfg(endpoint=server.url, model="decider-31b"))
            self.assertEqual(server.requests[0]["model"], "decider-31b")
        finally:
            server.stop()

    def test_engine_name_is_stripped(self):
        server = FakeServer()
        try:
            run_decision(
                "s", QUESTIONS, engine="  selfhosted  ",
                cfg=_cfg(endpoint=server.url),
            )
            self.assertEqual(len(server.requests), 1)
        finally:
            server.stop()

    def test_empty_answers_object_is_accepted(self):
        server = FakeServer(body={"answers": {}})
        try:
            out = run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
        finally:
            server.stop()
        self.assertEqual(out["answers"], {})

    def test_null_answers_is_malformed(self):
        server = FakeServer(body={"answers": None})
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
            self.assertIn("malformed SystemOne response", str(ctx.exception))
        finally:
            server.stop()

    def test_http_error_body_is_truncated(self):
        server = FakeServer(status=500, body={"detail": "x" * 500})
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=_cfg(endpoint=server.url))
        finally:
            server.stop()
        message = str(ctx.exception)
        self.assertIn("HTTP 500", message)
        self.assertLessEqual(message.count("x"), 200)

    def test_bytes_state_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(b"bytes", QUESTIONS, cfg=_cfg())
        self.assertIn("got bytes", str(ctx.exception))


class RegistryTests(unittest.TestCase):
    def test_vision_capable_models_are_sorted_and_complete(self):
        from weigh_n_decide.registry import vision_capable_models

        names = vision_capable_models()
        self.assertEqual(
            names,
            ["clef", "clef-flash", "decider-12b-vision", "decider-2b",
             "decider-31b", "decider-4b"],
        )

    def test_registry_entries_are_consistent(self):
        from weigh_n_decide.config import ENGINES
        from weigh_n_decide.registry import DEFAULT_MODEL, MODELS

        self.assertIn(DEFAULT_MODEL, MODELS)
        for name, info in MODELS.items():
            with self.subTest(model=name):
                self.assertIn(info.engine, ENGINES)
                self.assertIn(info.convention, (None, "assets", "images"))
                if info.max_images is not None:
                    self.assertGreater(info.max_images, 0)
                    self.assertIs(info.vision, True)
                if info.vision is False:
                    self.assertIsNone(info.convention)

    def test_vision_capable_models_is_a_fresh_copy(self):
        from weigh_n_decide.registry import vision_capable_models

        names = vision_capable_models()
        names.clear()
        self.assertEqual(len(vision_capable_models()), 6)

    def test_resolve_model_explicit_engine_wins_for_slashed_ids(self):
        from weigh_n_decide.registry import resolve_model

        engine, name, info = resolve_model(
            "vendor/new-model", "selfhosted", "clef", "selfhosted"
        )
        self.assertEqual((engine, name), ("selfhosted", "vendor/new-model"))
        self.assertIsNone(info.vision)

    def test_resolve_model_bare_defaults(self):
        from weigh_n_decide.registry import resolve_model

        engine, name, info = resolve_model(None, None, "decider-2b", "selfhosted")
        self.assertEqual((engine, name), ("selfhosted", "decider-2b"))
        self.assertIs(info.vision, True)

    def test_resolve_model_slash_unknown_is_openrouter(self):
        from weigh_n_decide.registry import resolve_model

        engine, name, info = resolve_model("vendor/new-model", None, "clef", None)
        self.assertEqual((engine, name), ("openrouter", "vendor/new-model"))
        self.assertIsNone(info.vision)

    def test_resolve_model_blank_name_is_an_error(self):
        from weigh_n_decide.registry import resolve_model

        with self.assertRaises(ValueError):
            resolve_model("  ", "selfhosted", "clef", "selfhosted")

    def test_resolve_model_engine_is_lowercased_and_stripped(self):
        from weigh_n_decide.registry import resolve_model

        engine, name, _ = resolve_model("clef", "  OpenRouter ", "clef", "selfhosted")
        self.assertEqual((engine, name), ("openrouter", "clef"))

    def test_vision_models_document_an_image_convention(self):
        from weigh_n_decide.registry import MODELS

        for name, info in MODELS.items():
            if info.vision is True:
                self.assertIsNotNone(info.convention, name)

    def test_model_info_is_frozen(self):
        from dataclasses import FrozenInstanceError

        from weigh_n_decide.registry import ModelInfo

        info = ModelInfo(engine="selfhosted")
        with self.assertRaises(FrozenInstanceError):
            info.vision = True

    def test_configured_default_engine_overrides_slash_fallback(self):
        from weigh_n_decide.registry import resolve_model

        engine, name, _ = resolve_model(
            "vendor/new-model", None, "clef", "selfhosted"
        )
        self.assertEqual((engine, name), ("selfhosted", "vendor/new-model"))


class RegressionTests(unittest.TestCase):
    def test_env_engine_overrides_registry_engine_and_loads_env(self):
        from unittest.mock import patch

        server = FakeServer()
        try:
            with patch.dict(
                os.environ,
                {
                    "WEIGH_N_DECIDE_ENGINE": "openrouter",
                    "WEIGH_N_DECIDE_MODEL": "clef",
                    "WEIGH_N_DECIDE_OPENROUTER_URL": server.url,
                    "OPENROUTER_API_KEY": "sk-env",
                },
                clear=True,
            ):
                out = run_decision("s", QUESTIONS)  # cfg=None -> load_config
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(server.requests[0]["model"], "clef")
            self.assertIn("Authorization", server.headers[0])
        finally:
            server.stop()

    def test_bad_fallback_does_not_block_a_live_primary(self):
        server = FakeServer()
        try:
            out = run_decision(
                "s", QUESTIONS, cfg=_cfg(endpoint=server.url, fallbacks=("typo-model",))
            )
            self.assertEqual(out["answers"], ANSWERS)
            self.assertEqual(len(server.requests), 1)
        finally:
            server.stop()

    def test_model_only_fallback_with_colon_suffix(self):
        server = FakeServer()
        try:
            cfg = _cfg(
                endpoint=None,
                openrouter_url=server.url,
                openrouter_key="k",
                fallbacks=("inception/mercury-decide:free",),
            )
            run_decision("s", QUESTIONS, cfg=cfg)
            self.assertEqual(
                server.requests[0]["model"], "inception/mercury-decide:free"
            )
        finally:
            server.stop()

    def test_fallback_parser_keeps_model_ids_with_slashes(self):
        from weigh_n_decide.engines import _parse_fallback

        self.assertEqual(
            _parse_fallback("inception/mercury-decide:free"),
            (None, "inception/mercury-decide:free"),
        )
        self.assertEqual(_parse_fallback("openrouter:clef"), ("openrouter", "clef"))
        self.assertEqual(_parse_fallback(":clef"), (None, "clef"))
        self.assertEqual(
            _parse_fallback("carrier-pigeon:clef"), ("carrier-pigeon", "clef")
        )

    def test_endpoint_already_containing_systemone_is_not_doubled(self):
        server = FakeServer()
        try:
            run_decision(
                "s", QUESTIONS, cfg=_cfg(endpoint=server.url + "/v1/systemone")
            )
            self.assertEqual(server.paths[0], "/v1/systemone")
        finally:
            server.stop()

    def test_non_finite_state_is_rejected(self):
        with self.assertRaises(EngineError):
            run_decision(float("nan"), QUESTIONS, cfg=_cfg())
        with self.assertRaises(EngineError):
            run_decision({"x": [float("inf")]}, QUESTIONS, cfg=_cfg())

    def test_unsupported_image_candidate_never_loads_images(self):
        from unittest.mock import patch

        with patch("weigh_n_decide.engines.load_image") as loader:
            with self.assertRaises(EngineError) as ctx:
                run_decision(
                    "s",
                    QUESTIONS,
                    images=[{"image": "http://example.invalid/a.png"}],
                    model="decider-12b",
                    cfg=_cfg(endpoint="http://127.0.0.1:1"),
                )
        self.assertIn("no image support", str(ctx.exception))
        loader.assert_not_called()

    def test_total_image_bytes_are_capped(self):
        from unittest.mock import patch

        with patch("weigh_n_decide.engines.load_image", return_value=("image/png", b"123456")):
            with patch("weigh_n_decide.engines.MAX_TOTAL_IMAGE_BYTES", 5):
                with self.assertRaises(EngineError) as ctx:
                    run_decision(
                        "s",
                        QUESTIONS,
                        images=[PNG_DATA_URI],
                        model="clef",
                        cfg=_cfg(endpoint="http://127.0.0.1:1"),
                    )
        self.assertIn("total limit", str(ctx.exception))

    def test_unknown_image_spec_field_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s",
                QUESTIONS,
                images=[{"image": PNG_DATA_URI, "reff": "x"}],
                model="clef",
                cfg=_cfg(endpoint="http://127.0.0.1:1"),
            )
        self.assertIn("unknown field", str(ctx.exception))

    def test_whitespace_image_ref_is_rejected(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s",
                QUESTIONS,
                images=[{"image": PNG_DATA_URI, "ref": "   "}],
                model="clef",
                cfg=_cfg(endpoint="http://127.0.0.1:1"),
            )
        self.assertIn("'ref' must be a non-empty string", str(ctx.exception))

    def test_image_url_policy_public_blocks_loopback(self):
        server = FakeServer()
        try:
            with self.assertRaises(EngineError) as ctx:
                run_decision(
                    "s",
                    QUESTIONS,
                    images=[server.url + "/pic.png"],
                    model="clef",
                    cfg=_cfg(
                        endpoint=server.url,
                        image_url_policy="public",
                    ),
                )
            self.assertIn("non-public address", str(ctx.exception))
            self.assertEqual(server.requests, [])
        finally:
            server.stop()

    def test_timeout_is_clamped_by_the_overall_deadline(self):
        import httpx
        from unittest import mock

        seen = {}

        def fake_post(url, **kwargs):
            seen["timeout"] = kwargs["timeout"]
            raise httpx.ConnectError("stop")

        cfg = _cfg(
            endpoint="http://127.0.0.1:1",
            deadline=0.05,
            fallbacks=("openrouter:clef",),
            openrouter_key="k",
            openrouter_url="http://127.0.0.1:2",
        )
        with mock.patch("weigh_n_decide.engines._CLIENT.post", side_effect=fake_post):
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
        self.assertLessEqual(seen["timeout"], 0.05)
        self.assertIn("all engine attempts failed", str(ctx.exception))

    def test_choice_option_ids_must_be_non_empty_strings(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s",
                {"x": {"type": "choice", "criteria": {1: "one"}}},
                cfg=_cfg(),
            )
        self.assertIn("option ids must be non-empty strings", str(ctx.exception))

    def test_endpoint_must_be_an_http_url(self):
        with self.assertRaises(EngineError) as ctx:
            run_decision("s", QUESTIONS, cfg=_cfg(endpoint="ftp://decide"))
        self.assertIn("must be an http(s) URL", str(ctx.exception))

    def test_openrouter_url_must_be_an_http_url(self):
        cfg = _cfg(
            endpoint=None,
            openrouter_url="ftp://openrouter",
            openrouter_key="k",
        )
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s", QUESTIONS, model="inception/mercury-decide", cfg=cfg
            )
        self.assertIn("must be an http(s) URL", str(ctx.exception))

    def test_deadline_before_image_loading(self):
        from unittest import mock

        cfg = _cfg(endpoint="http://127.0.0.1:1", deadline=1.0)
        with mock.patch(
            "weigh_n_decide.engines.time.monotonic", side_effect=[0.0, 10.0]
        ):
            with self.assertRaises(EngineError) as ctx:
                run_decision(
                    "s", QUESTIONS, images=[PNG_DATA_URI], model="clef", cfg=cfg
                )
        self.assertIn("deadline exceeded before image loading", str(ctx.exception))

    def test_deadline_stops_the_attempt_loop(self):
        from unittest import mock

        cfg = _cfg(
            endpoint="http://127.0.0.1:1",
            deadline=1.0,
            fallbacks=("openrouter:clef",),
            openrouter_key="k",
            openrouter_url="http://127.0.0.1:2",
        )
        with mock.patch(
            "weigh_n_decide.engines.time.monotonic", side_effect=[0.0, 10.0]
        ):
            with self.assertRaises(EngineError) as ctx:
                run_decision("s", QUESTIONS, cfg=cfg)
        self.assertIn("overall deadline exceeded", str(ctx.exception))

    def test_unknown_image_url_policy_surfaces(self):
        cfg = _cfg(
            endpoint="http://127.0.0.1:1", image_url_policy="sometimes"
        )
        with self.assertRaises(EngineError) as ctx:
            run_decision(
                "s",
                QUESTIONS,
                images=["http://example.com/a.png"],
                model="clef",
                cfg=cfg,
            )
        self.assertIn("unknown image URL policy", str(ctx.exception))

    def test_deadline_during_image_loading(self):
        from unittest import mock

        cfg = _cfg(endpoint="http://127.0.0.1:1", deadline=1.0)
        with mock.patch(
            "weigh_n_decide.engines.time.monotonic",
            side_effect=[0.0, 0.0, 10.0],
        ):
            with self.assertRaises(EngineError) as ctx:
                run_decision(
                    "s", QUESTIONS, images=[PNG_DATA_URI], model="clef", cfg=cfg
                )
        self.assertIn("deadline exceeded while loading images", str(ctx.exception))

if __name__ == "__main__":


    unittest.main()