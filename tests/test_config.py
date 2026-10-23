"""Config tests: env-driven defaults, fallback parsing, and invalid inputs."""

import os
import unittest
from unittest.mock import patch

from weigh_n_decide.config import (
    DEFAULT_OPENROUTER_URL,
    Config,
    _parse_fallbacks,
    load_config,
)
from weigh_n_decide.registry import DEFAULT_MODEL

_ENV_KEYS = (
    "WEIGH_N_DECIDE_ENGINE",
    "WEIGH_N_DECIDE_ENDPOINT",
    "WEIGH_N_DECIDE_MODEL",
    "WEIGH_N_DECIDE_FALLBACKS",
    "WEIGH_N_DECIDE_OPENROUTER_URL",
    "WEIGH_N_DECIDE_TIMEOUT",
    "WEIGH_N_DECIDE_DEADLINE",
    "WEIGH_N_DECIDE_IMAGE_URL_POLICY",
    "OPENROUTER_API_KEY",
)


def _env(**values):
    return patch.dict(os.environ, values, clear=True)


class LoadConfigTests(unittest.TestCase):
    def test_defaults_when_no_env(self):
        with _env():
            cfg = load_config()
        self.assertIsNone(cfg.engine)
        self.assertIsNone(cfg.endpoint)
        self.assertEqual(cfg.model, DEFAULT_MODEL)
        self.assertEqual(cfg.fallbacks, ())
        self.assertEqual(cfg.openrouter_url, DEFAULT_OPENROUTER_URL)
        self.assertIsNone(cfg.openrouter_key)
        self.assertEqual(cfg.timeout, 60.0)
        self.assertEqual(cfg.deadline, 120.0)
        self.assertEqual(cfg.image_url_policy, "public")

    def test_engine_env(self):
        with _env(WEIGH_N_DECIDE_ENGINE="openrouter"):
            self.assertEqual(load_config().engine, "openrouter")

    def test_unknown_engine_env_is_preserved_for_call_time_error(self):
        with _env(WEIGH_N_DECIDE_ENGINE="carrier-pigeon"):
            self.assertEqual(load_config().engine, "carrier-pigeon")

    def test_engine_env_is_lowercased_and_stripped(self):
        with _env(WEIGH_N_DECIDE_ENGINE="  OpenRouter "):
            self.assertEqual(load_config().engine, "openrouter")

    def test_endpoint_env(self):
        with _env(WEIGH_N_DECIDE_ENDPOINT="http://decide:8080"):
            self.assertEqual(load_config().endpoint, "http://decide:8080")

    def test_model_env(self):
        with _env(WEIGH_N_DECIDE_MODEL="decider-31b"):
            self.assertEqual(load_config().model, "decider-31b")

    def test_env_values_are_stripped(self):
        with _env(
            WEIGH_N_DECIDE_MODEL="  clef ",
            WEIGH_N_DECIDE_OPENROUTER_URL="  https://or.test/x  ",
            WEIGH_N_DECIDE_TIMEOUT=" 5 ",
        ):
            cfg = load_config()
        self.assertEqual(cfg.model, "clef")
        self.assertEqual(cfg.openrouter_url, "https://or.test/x")
        self.assertEqual(cfg.timeout, 5.0)

    def test_non_positive_or_non_finite_timeout_uses_default(self):
        for raw in ("0", "-5", "nan", "inf", "-inf", "1e999"):
            with _env(WEIGH_N_DECIDE_TIMEOUT=raw):
                self.assertEqual(load_config().timeout, 60.0, raw)

    def test_blank_env_values_use_defaults(self):
        with _env(
            WEIGH_N_DECIDE_ENGINE="   ",
            WEIGH_N_DECIDE_MODEL="   ",
            WEIGH_N_DECIDE_ENDPOINT="   ",
            WEIGH_N_DECIDE_OPENROUTER_URL="   ",
            WEIGH_N_DECIDE_TIMEOUT="   ",
            OPENROUTER_API_KEY="   ",
        ):
            cfg = load_config()
        self.assertIsNone(cfg.engine)
        self.assertEqual(cfg.model, DEFAULT_MODEL)
        self.assertIsNone(cfg.endpoint)
        self.assertEqual(cfg.openrouter_url, DEFAULT_OPENROUTER_URL)
        self.assertIsNone(cfg.openrouter_key)
        self.assertEqual(cfg.timeout, 60.0)
        self.assertEqual(cfg.deadline, 120.0)
        self.assertEqual(cfg.image_url_policy, "public")

    def test_endpoint_and_key_are_stripped(self):
        with _env(
            WEIGH_N_DECIDE_ENDPOINT="  http://decide:8080  ",
            OPENROUTER_API_KEY="  sk-secret  ",
        ):
            cfg = load_config()
        self.assertEqual(cfg.endpoint, "http://decide:8080")
        self.assertEqual(cfg.openrouter_key, "sk-secret")

    def test_openrouter_url_env(self):
        with _env(WEIGH_N_DECIDE_OPENROUTER_URL="https://or.test/api/v1/systemone"):
            self.assertEqual(
                load_config().openrouter_url, "https://or.test/api/v1/systemone"
            )

    def test_openrouter_key_env(self):
        with _env(OPENROUTER_API_KEY="sk-secret"):
            self.assertEqual(load_config().openrouter_key, "sk-secret")

    def test_timeout_env(self):
        with _env(WEIGH_N_DECIDE_TIMEOUT="5.5"):
            self.assertEqual(load_config().timeout, 5.5)

    def test_invalid_timeout_env_uses_default(self):
        with _env(WEIGH_N_DECIDE_TIMEOUT="soon"):
            self.assertEqual(load_config().timeout, 60.0)

    def test_deadline_env(self):
        with _env(WEIGH_N_DECIDE_DEADLINE="15.5"):
            self.assertEqual(load_config().deadline, 15.5)
        for raw in ("0", "-1", "nan", "inf", "soon"):
            with _env(WEIGH_N_DECIDE_DEADLINE=raw):
                self.assertEqual(load_config().deadline, 120.0, raw)

    def test_image_url_policy_env(self):
        with _env(WEIGH_N_DECIDE_IMAGE_URL_POLICY="  ANY "):
            self.assertEqual(load_config().image_url_policy, "any")
        with _env(WEIGH_N_DECIDE_IMAGE_URL_POLICY="   "):
            self.assertEqual(load_config().image_url_policy, "public")

    def test_fallbacks_env_parsed_and_stripped(self):
        with _env(WEIGH_N_DECIDE_FALLBACKS="openrouter:clef, decider-2b ,"):
            self.assertEqual(load_config().fallbacks, ("openrouter:clef", "decider-2b"))

    def test_empty_fallbacks_env(self):
        with _env(WEIGH_N_DECIDE_FALLBACKS="  , ,"):
            self.assertEqual(load_config().fallbacks, ())


class ParseFallbacksTests(unittest.TestCase):
    def test_none_and_empty(self):
        self.assertEqual(_parse_fallbacks(None), ())
        self.assertEqual(_parse_fallbacks(""), ())
        self.assertEqual(_parse_fallbacks("   "), ())

    def test_splits_and_strips(self):
        self.assertEqual(_parse_fallbacks("a , b"), ("a", "b"))
        self.assertEqual(
            _parse_fallbacks("openrouter:inception/mercury-decide:free"),
            ("openrouter:inception/mercury-decide:free",),
        )

    def test_interior_empty_entries_are_dropped(self):
        self.assertEqual(_parse_fallbacks("a,,b, ,c"), ("a", "b", "c"))

    def test_engine_only_and_model_only_entries_survive(self):
        self.assertEqual(
            _parse_fallbacks(" openrouter: , :clef ,inception/mercury-decide:free"),
            ("openrouter:", ":clef", "inception/mercury-decide:free"),
        )


class ConfigDefaultsTests(unittest.TestCase):
    def test_dataclass_defaults(self):
        cfg = Config()
        self.assertIsNone(cfg.engine)
        self.assertIsNone(cfg.endpoint)
        self.assertEqual(cfg.model, DEFAULT_MODEL)
        self.assertEqual(cfg.fallbacks, ())
        self.assertEqual(cfg.openrouter_url, DEFAULT_OPENROUTER_URL)
        self.assertIsNone(cfg.openrouter_key)
        self.assertEqual(cfg.timeout, 60.0)
        self.assertEqual(cfg.deadline, 120.0)
        self.assertEqual(cfg.image_url_policy, "public")

    def test_config_is_frozen(self):
        import dataclasses

        cfg = Config()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg.timeout = 5.0


if __name__ == "__main__":
    unittest.main()
