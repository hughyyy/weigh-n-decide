"""Environment-driven configuration with sane defaults."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .registry import DEFAULT_MODEL

ENGINES = ("selfhosted", "openrouter")
# OpenRouter's System One endpoint is not yet documented; this route is the
# working default and WEIGH_N_DECIDE_OPENROUTER_URL overrides it (open
# question in README).
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/systemone"

DEFAULT_TIMEOUT = 60.0
DEFAULT_DEADLINE = 120.0
DEFAULT_IMAGE_URL_POLICY = "public"


@dataclass(frozen=True)
class Config:
    # None means "use the model's own engine"; an explicit value (per call or
    # WEIGH_N_DECIDE_ENGINE) overrides the registry.
    engine: str | None = None
    endpoint: str | None = None
    model: str = DEFAULT_MODEL
    fallbacks: tuple[str, ...] = ()
    openrouter_url: str = DEFAULT_OPENROUTER_URL
    openrouter_key: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    deadline: float = DEFAULT_DEADLINE
    image_url_policy: str = DEFAULT_IMAGE_URL_POLICY


def load_config() -> Config:
    engine = _env_text("WEIGH_N_DECIDE_ENGINE")
    return Config(
        engine=engine.lower() if engine else None,
        endpoint=_env_text("WEIGH_N_DECIDE_ENDPOINT"),
        model=_env_text("WEIGH_N_DECIDE_MODEL") or DEFAULT_MODEL,
        fallbacks=_parse_fallbacks(os.environ.get("WEIGH_N_DECIDE_FALLBACKS")),
        openrouter_url=_env_text("WEIGH_N_DECIDE_OPENROUTER_URL")
        or DEFAULT_OPENROUTER_URL,
        openrouter_key=_env_text("OPENROUTER_API_KEY"),
        timeout=_parse_positive(
            os.environ.get("WEIGH_N_DECIDE_TIMEOUT"), DEFAULT_TIMEOUT
        ),
        deadline=_parse_positive(
            os.environ.get("WEIGH_N_DECIDE_DEADLINE"), DEFAULT_DEADLINE
        ),
        image_url_policy=(
            _env_text("WEIGH_N_DECIDE_IMAGE_URL_POLICY") or DEFAULT_IMAGE_URL_POLICY
        ).lower(),
    )


def _env_text(name: str) -> str | None:
    """An env value with surrounding whitespace removed, or None when blank."""
    return (os.environ.get(name) or "").strip() or None


def _parse_positive(raw: str | None, default: float) -> float:
    """Seconds; blank, non-numeric, non-finite, and non-positive fall back."""
    try:
        value = float((raw or "").strip())
    except ValueError:
        return default
    return value if math.isfinite(value) and value > 0 else default


def _parse_fallbacks(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())