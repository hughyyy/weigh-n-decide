"""Environment-driven configuration with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .registry import DEFAULT_MODEL

ENGINES = ("selfhosted", "openrouter")
# OpenRouter's System One endpoint is not yet documented; this route is the
# working default and WEIGH_N_DECIDE_OPENROUTER_URL overrides it (open
# question in README).
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/systemone"


@dataclass(frozen=True)
class Config:
    engine: str = "selfhosted"
    endpoint: str | None = None
    model: str = DEFAULT_MODEL
    fallbacks: tuple[str, ...] = ()
    openrouter_url: str = DEFAULT_OPENROUTER_URL
    openrouter_key: str | None = None
    timeout: float = 60.0


def load_config() -> Config:
    engine = (os.environ.get("WEIGH_N_DECIDE_ENGINE") or "selfhosted").strip().lower()
    if engine not in ENGINES:
        engine = "selfhosted"
    model = (os.environ.get("WEIGH_N_DECIDE_MODEL") or DEFAULT_MODEL).strip()
    raw_timeout = (os.environ.get("WEIGH_N_DECIDE_TIMEOUT") or "").strip()
    try:
        timeout = float(raw_timeout) if raw_timeout else 60.0
    except ValueError:
        timeout = 60.0
    return Config(
        engine=engine,
        endpoint=(os.environ.get("WEIGH_N_DECIDE_ENDPOINT") or None),
        model=model,
        fallbacks=_parse_fallbacks(os.environ.get("WEIGH_N_DECIDE_FALLBACKS")),
        openrouter_url=(
            os.environ.get("WEIGH_N_DECIDE_OPENROUTER_URL") or DEFAULT_OPENROUTER_URL
        ).strip(),
        openrouter_key=os.environ.get("OPENROUTER_API_KEY") or None,
        timeout=timeout,
    )


def _parse_fallbacks(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())