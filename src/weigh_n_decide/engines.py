"""SystemOne engines: request building, HTTP dispatch, fallback chains.

One wire format (Jev/SystemOne, `POST /v1/systemone`) serves every backend;
engines differ only in endpoint resolution and auth. Inputs are validated
strictly — the server never invents request or response fields. Response
answers are passed through unchanged: validating individual answer shapes
awaits confirmed card parity across backends (README open question).
"""

from __future__ import annotations

import atexit
import base64
import math
import time

import httpx

from .config import ENGINES, load_config
from .images import ImageInputError, MAX_TOTAL_IMAGE_BYTES, load_image
from .registry import ModelInfo, resolve_model, vision_capable_models

__all__ = ["EngineError", "run_decision"]

_QUESTION_TYPES = ("noul", "choice", "score")
_QUESTION_FIELDS = {"type", "instructions", "criteria"}
_IMAGE_SPEC_FIELDS = {"image", "ref"}
_IMAGE_FORMATS = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg"}

# One client per process: TLS sessions and connection pools are reused across
# calls. httpx.Client is thread-safe, and the MCP SDK runs sync tools on
# worker threads.
_CLIENT = httpx.Client()
atexit.register(_CLIENT.close)


class EngineError(ValueError):
    """Raised for invalid requests, unconfigured engines, and failed dispatch."""


# --------------------------------------------------------------------------
# Input validation

def _validate_state(state: object) -> None:
    if state is None:
        raise EngineError("state is required (a string or JSON value)")
    if isinstance(state, bool) or isinstance(state, (int, str)):
        return
    if isinstance(state, float):
        if not math.isfinite(state):
            raise EngineError("state contains a non-finite number")
        return
    if isinstance(state, (list, dict)):
        _validate_finite(state)
        return
    raise EngineError(
        f"state must be a string or JSON value (got {type(state).__name__})"
    )


def _validate_finite(root: object) -> None:
    stack = [root]
    seen: set[int] = set()
    while stack:
        value = stack.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise EngineError("state contains a non-finite number")
        if isinstance(value, (list, dict)) and id(value) not in seen:
            seen.add(id(value))
            stack.extend(value.values() if isinstance(value, dict) else value)


def _validate_questions(questions: object) -> None:
    if not isinstance(questions, dict) or not questions:
        raise EngineError("questions must be a non-empty object of question id -> question")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise EngineError("question ids must be non-empty strings")
        if not isinstance(question, dict):
            raise EngineError(f"question {qid!r} must be an object")
        unknown = sorted(map(str, set(question) - _QUESTION_FIELDS))
        if unknown:
            raise EngineError(
                f"question {qid!r}: unknown field(s) {', '.join(unknown)} "
                f"(allowed: type, instructions, criteria)"
            )
        qtype = question.get("type")
        if qtype not in _QUESTION_TYPES:
            raise EngineError(
                f"question {qid!r}: type must be one of "
                f"{'/'.join(_QUESTION_TYPES)} (got {qtype!r})"
            )
        instructions = question.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise EngineError(f"question {qid!r}: instructions must be a string")
        criteria = question.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise EngineError(
                    f"question {qid!r}: choice needs criteria as a non-empty "
                    f"object of option id -> description"
                )
            for oid, desc in criteria.items():
                if not isinstance(oid, str) or not oid:
                    raise EngineError(
                        f"question {qid!r}: criteria option ids must be non-empty strings"
                    )
                if not isinstance(desc, str):
                    raise EngineError(
                        f"question {qid!r}: criteria option {oid!r} description must be a string"
                    )
        elif qtype == "score":
            if not isinstance(criteria, list) or not criteria:
                raise EngineError(
                    f"question {qid!r}: score needs criteria as a non-empty "
                    f"list of ordered legend descriptions"
                )
            if not all(isinstance(item, str) for item in criteria):
                raise EngineError(f"question {qid!r}: legend entries must be strings")
        else:  # noul
            if criteria is None:
                continue
            if not isinstance(criteria, dict):
                raise EngineError(
                    f"question {qid!r}: noul criteria must be an object with "
                    f"optional 'true'/'false' descriptions"
                )
            for key, desc in criteria.items():
                if key not in ("true", "false"):
                    raise EngineError(
                        f"question {qid!r}: noul criteria keys must be 'true'/'false' "
                        f"(got {key!r})"
                    )
                if not isinstance(desc, str):
                    raise EngineError(f"question {qid!r}: criteria {key!r} must be a string")


class _ImageSpec:
    __slots__ = ("index", "ref", "source")

    def __init__(self, index: int, ref: str, source: str):
        self.index = index
        self.ref = ref
        self.source = source


class _Image:
    __slots__ = ("ref", "media", "b64", "fmt")

    def __init__(self, ref: str, media: str, b64: str, fmt: str):
        self.ref = ref
        self.media = media
        self.b64 = b64
        self.fmt = fmt


def _image_specs(images: object) -> list[_ImageSpec]:
    """Validate image entries without touching the filesystem or network."""
    if images is None:
        return []
    if not isinstance(images, list) or not images:
        raise EngineError("images must be a non-empty list when given")
    specs: list[_ImageSpec] = []
    for index, entry in enumerate(images):
        if isinstance(entry, str):
            spec: dict = {"image": entry}
        elif isinstance(entry, dict):
            spec = entry
        else:
            raise EngineError(
                f"images[{index}] must be a string or an object with an 'image' field"
            )
        unknown = sorted(map(str, set(spec) - _IMAGE_SPEC_FIELDS))
        if unknown:
            raise EngineError(
                f"images[{index}]: unknown field(s) {', '.join(unknown)} "
                f"(allowed: image, ref)"
            )
        source = spec.get("image")
        if not isinstance(source, str) or not source.strip():
            raise EngineError(f"images[{index}]: 'image' must be a non-empty string")
        ref = spec.get("ref")
        if ref is None:
            ref = f"image_{index}"
        elif not isinstance(ref, str) or not ref.strip():
            raise EngineError(f"images[{index}]: 'ref' must be a non-empty string")
        else:
            ref = ref.strip()
        specs.append(_ImageSpec(index, ref, source))
    refs = [spec.ref for spec in specs]
    if len(set(refs)) != len(refs):
        raise EngineError(f"duplicate image refs: {refs}")
    return specs


def _load_images(
    specs: list[_ImageSpec], *, url_policy: str, timeout: float, deadline_at: float
) -> list[_Image]:
    resolved: list[_Image] = []
    total = 0
    for spec in specs:
        remaining = deadline_at - time.monotonic()
        if remaining <= 0:
            raise EngineError("overall deadline exceeded while loading images")
        try:
            media, data = load_image(
                spec.source,
                url_policy=url_policy,
                timeout=min(timeout, remaining),
            )
        except ImageInputError as exc:
            raise EngineError(
                f"images[{spec.index}] ({spec.ref}): {exc}"
            ) from exc
        total += len(data)
        if total > MAX_TOTAL_IMAGE_BYTES:
            raise EngineError(
                f"images exceed the "
                f"{MAX_TOTAL_IMAGE_BYTES // (1024 * 1024)} MiB total limit"
            )
        fmt = _IMAGE_FORMATS.get(media.rsplit("/", 1)[-1].lower())
        if fmt is None:
            raise EngineError(
                f"images[{spec.index}] ({spec.ref}): unsupported format {media!r} "
                f"(png and jpeg)"
            )
        resolved.append(_Image(spec.ref, media, base64.b64encode(data).decode(), fmt))
    return resolved


# --------------------------------------------------------------------------
# Request building

def _build_payload(model: str, state: object, questions: dict, imgs: list[_Image], info: ModelInfo) -> dict:
    payload: dict = {"model": model, "state": state, "questions": questions}
    if not imgs:
        return payload
    if info.convention == "assets":
        payload["assets"] = {
            img.ref: {"type": "image", "format": img.fmt, "data_base64": img.b64}
            for img in imgs
        }
    elif info.convention == "images":
        payload["images"] = [f"data:{img.media};base64,{img.b64}" for img in imgs]
    else:
        raise EngineError(
            f"model {model!r} has no documented image convention; "
            f"vision-capable models: {', '.join(vision_capable_models())}"
        )
    return payload


# --------------------------------------------------------------------------
# Dispatch

def _require_http_url(url: str, name: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        raise EngineError(f"{name} must be an http(s) URL (got {url!r})")
    return url


def _resolve_url(engine: str, cfg) -> tuple[str, dict[str, str]]:
    if engine == "selfhosted":
        if not cfg.endpoint:
            raise EngineError(
                "no self-hosted endpoint configured; set WEIGH_N_DECIDE_ENDPOINT "
                "to a /v1/systemone base URL (SGLang Clef or decider.serve)"
            )
        endpoint = _require_http_url(
            cfg.endpoint.rstrip("/"), "WEIGH_N_DECIDE_ENDPOINT"
        )
        url = (
            endpoint
            if endpoint.lower().endswith("/v1/systemone")
            else endpoint + "/v1/systemone"
        )
        return url, {}
    if engine == "openrouter":
        if not cfg.openrouter_key:
            raise EngineError(
                "OPENROUTER_API_KEY is not set (required for the openrouter engine)"
            )
        if not cfg.openrouter_url:
            raise EngineError(
                "WEIGH_N_DECIDE_OPENROUTER_URL is empty (needs the full "
                "System One endpoint URL)"
            )
        return (
            _require_http_url(
                cfg.openrouter_url, "WEIGH_N_DECIDE_OPENROUTER_URL"
            ),
            {"Authorization": f"Bearer {cfg.openrouter_key}"},
        )
    raise EngineError(f"unknown engine {engine!r} (engines: {'/'.join(ENGINES)})")


def _call_systemone(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    try:
        resp = _CLIENT.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
        raise EngineError(f"request to {url} failed: {exc}") from exc
    except (ValueError, OverflowError) as exc:
        # A hand-built Config can carry an invalid timeout; httpx rejects it
        # before sending. Keep the EngineError contract for callers.
        raise EngineError(f"request to {url} failed: {exc}") from exc
    except TypeError as exc:
        # A non-JSON state or question value fails during encoding; the
        # request contract says JSON, so surface it as a request error.
        raise EngineError(f"request to {url} failed: {exc}") from exc
    if resp.status_code != 200:
        body = resp.text[:200]
        raise EngineError(f"{url} returned HTTP {resp.status_code}: {body}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise EngineError(f"response from {url} was not JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise EngineError(
            f"malformed SystemOne response from {url}: expected an object "
            f"with an 'answers' object"
        )
    # Answers are passed through unvalidated; per-backend answer-field parity
    # is still an open question (README).
    return {
        "model": data.get("model", payload["model"]),
        "answers": data["answers"],
        "usage": data.get("usage"),
    }


def _parse_fallback(entry: str) -> tuple[str | None, str | None]:
    """Split one fallback entry into (engine, model).

    `engine:model` is split only when the prefix is a known engine or an
    engine-like bare name. A model id whose pre-colon prefix contains `/`
    stays intact, so OpenRouter ids such as `inception/mercury-decide:free`
    work without an engine prefix.
    """
    entry = entry.strip()
    if not entry:
        return None, None
    if ":" not in entry:
        return None, entry
    head, rest = entry.split(":", 1)
    head = head.strip()
    model = rest.strip() or None
    if not head:
        return None, model
    if head.lower() in ENGINES or "/" not in head:
        return head, model
    return None, entry


def _candidates(
    model: str | None, engine: str | None, cfg
) -> tuple[list[tuple[str, str, ModelInfo]], list[str]]:
    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str, ModelInfo]] = []
    errors: list[str] = []

    def add(model_name: str | None, engine_name: str | None, *, required: bool) -> None:
        try:
            resolved = resolve_model(model_name, engine_name, cfg.model, cfg.engine)
        except ValueError as exc:
            if required:
                raise EngineError(str(exc)) from exc
            errors.append(str(exc))
            return
        key = (resolved[0], resolved[1])
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)

    add(model, engine, required=True)
    for fallback in cfg.fallbacks:
        fb_engine, fb_model = _parse_fallback(fallback)
        add(fb_model, fb_engine, required=False)
    return candidates, errors


def _image_support_error(info: ModelInfo, count: int) -> str | None:
    if info.vision is False:
        return "no image support"
    if info.convention is None:
        return (
            "no documented image convention "
            f"(vision-capable: {', '.join(vision_capable_models())})"
        )
    if info.max_images is not None and count > info.max_images:
        return f"max {info.max_images} images (got {count})"
    return None


def run_decision(
    state: object,
    questions: object,
    images: object = None,
    model: str | None = None,
    engine: str | None = None,
    cfg=None,
) -> dict:
    """Validate inputs, then dispatch through the engine/fallback chain.

    Returns the normalized SystemOne response: {"model", "answers", "usage"}.
    Raises EngineError with every failed attempt when nothing succeeds.
    """
    cfg = cfg or load_config()
    _validate_state(state)
    _validate_questions(questions)
    specs = _image_specs(images)

    started = time.monotonic()
    attempts, errors = _candidates(model, engine, cfg)

    supported: set[int] = set()
    if specs:
        for index, (eng, name, info) in enumerate(attempts):
            problem = _image_support_error(info, len(specs))
            if problem is None:
                supported.add(index)
            else:
                errors.append(f"{eng}:{name}: {problem}")
        if not supported:
            raise EngineError("all engine attempts failed — " + "; ".join(errors))

    imgs: list[_Image] = []
    if specs:
        remaining = cfg.deadline - (time.monotonic() - started)
        if remaining <= 0:
            raise EngineError("overall deadline exceeded before image loading")
        imgs = _load_images(
            specs,
            url_policy=cfg.image_url_policy,
            timeout=cfg.timeout,
            deadline_at=started + cfg.deadline,
        )

    for index, (eng, name, info) in enumerate(attempts):
        if specs and index not in supported:
            continue
        remaining = cfg.deadline - (time.monotonic() - started)
        if remaining <= 0:
            errors.append(f"{eng}:{name}: overall deadline exceeded")
            break
        try:
            payload = _build_payload(name, state, questions, imgs, info)
            url, headers = _resolve_url(eng, cfg)
            return _call_systemone(
                url, headers, payload, min(cfg.timeout, remaining)
            )
        except EngineError as exc:
            errors.append(f"{eng}:{name}: {exc}")
    raise EngineError("all engine attempts failed — " + "; ".join(errors))
