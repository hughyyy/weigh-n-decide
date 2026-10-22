"""SystemOne engines: request building, HTTP dispatch, fallback chains.

One wire format (Jev/SystemOne, `POST /v1/systemone`) serves every backend;
engines differ only in endpoint resolution and auth. Inputs are validated
strictly — the server never invents request or response fields.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import httpx

from .images import ImageInputError, load_image
from .registry import ModelInfo, resolve_model, vision_capable_models

if TYPE_CHECKING:
    from .config import Config

__all__ = ["EngineError", "run_decision"]

_QUESTION_TYPES = ("noul", "choice", "score")
_QUESTION_FIELDS = {"type", "instructions", "criteria"}
_IMAGE_FORMATS = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg"}


class EngineError(ValueError):
    """Raised for invalid requests, unconfigured engines, and failed dispatch."""


# --------------------------------------------------------------------------
# Input validation

def _validate_state(state: object) -> None:
    if state is None:
        raise EngineError("state is required (a string or JSON value)")
    if isinstance(state, bool) or isinstance(state, (int, float, str, list, dict)):
        return
    raise EngineError(
        f"state must be a string or JSON value (got {type(state).__name__})"
    )


def _validate_questions(questions: object) -> None:
    if not isinstance(questions, dict) or not questions:
        raise EngineError("questions must be a non-empty object of question id -> question")
    for qid, question in questions.items():
        if not isinstance(qid, str) or not qid:
            raise EngineError("question ids must be non-empty strings")
        if not isinstance(question, dict):
            raise EngineError(f"question {qid!r} must be an object")
        unknown = sorted(set(question) - _QUESTION_FIELDS)
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


class _Image:
    __slots__ = ("ref", "media", "b64", "fmt")

    def __init__(self, ref: str, media: str, b64: str, fmt: str):
        self.ref = ref
        self.media = media
        self.b64 = b64
        self.fmt = fmt


def _resolve_images(images: object) -> list[_Image]:
    if images is None:
        return []
    if not isinstance(images, list) or not images:
        raise EngineError("images must be a non-empty list when given")
    resolved: list[_Image] = []
    for index, entry in enumerate(images):
        if isinstance(entry, str):
            spec: dict = {"image": entry}
        elif isinstance(entry, dict):
            spec = entry
        else:
            raise EngineError(
                f"images[{index}] must be a string or an object with an 'image' field"
            )
        source = spec.get("image")
        if not isinstance(source, str) or not source:
            raise EngineError(f"images[{index}]: 'image' must be a non-empty string")
        ref = spec.get("ref")
        if ref is None:
            ref = f"image_{index}"
        if not isinstance(ref, str) or not ref:
            raise EngineError(f"images[{index}]: 'ref' must be a non-empty string")
        try:
            media, data = load_image(source)
        except ImageInputError as exc:
            raise EngineError(f"images[{index}] ({ref}): {exc}") from exc
        fmt = _IMAGE_FORMATS.get(media.rsplit("/", 1)[-1].lower())
        if fmt is None:
            raise EngineError(
                f"images[{index}] ({ref}): unsupported format {media!r} (png and jpeg)"
            )
        resolved.append(_Image(ref, media, base64.b64encode(data).decode(), fmt))
    refs = [img.ref for img in resolved]
    if len(set(refs)) != len(refs):
        raise EngineError(f"duplicate image refs: {refs}")
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

def _resolve_url(engine: str, cfg: "Config") -> tuple[str, dict[str, str]]:
    if engine == "selfhosted":
        if not cfg.endpoint:
            raise EngineError(
                "no self-hosted endpoint configured; set WEIGH_N_DECIDE_ENDPOINT "
                "to a /v1/systemone base URL (SGLang Clef or decider.serve)"
            )
        return cfg.endpoint.rstrip("/") + "/v1/systemone", {}
    if engine == "openrouter":
        if not cfg.openrouter_key:
            raise EngineError(
                "OPENROUTER_API_KEY is not set (required for the openrouter engine)"
            )
        return cfg.openrouter_url, {"Authorization": f"Bearer {cfg.openrouter_key}"}
    raise EngineError(f"unknown engine {engine!r} (engines: selfhosted, openrouter)")


def _call_systemone(url: str, headers: dict[str, str], payload: dict, timeout: float) -> dict:
    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError as exc:
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
    return {
        "model": data.get("model", payload["model"]),
        "answers": data["answers"],
        "usage": data.get("usage"),
    }


def _candidates(
    model: str | None, engine: str | None, cfg: "Config"
) -> list[tuple[str, str, ModelInfo]]:
    seen: set[tuple[str, str]] = set()
    candidates: list[tuple[str, str, ModelInfo]] = []

    def add(model_name: str | None, engine_name: str | None) -> None:
        try:
            resolved = resolve_model(model_name, engine_name, cfg.model, cfg.engine)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        key = (resolved[0], resolved[1])
        if key not in seen:
            seen.add(key)
            candidates.append(resolved)

    add(model, engine)
    for fallback in cfg.fallbacks:
        if ":" in fallback:
            fb_engine, fb_model = fallback.split(":", 1)
            add(fb_model.strip() or None, fb_engine.strip())
        else:
            add(fallback.strip() or None, None)
    return candidates


def run_decision(
    state: object,
    questions: object,
    images: object = None,
    model: str | None = None,
    engine: str | None = None,
    cfg: "Config | None" = None,
) -> dict:
    """Validate inputs, then dispatch through the engine/fallback chain.

    Returns the normalized SystemOne response: {"model", "answers", "usage"}.
    Raises EngineError with every failed attempt when nothing succeeds.
    """
    from .config import Config as _Config

    cfg = cfg or _Config()
    _validate_state(state)
    _validate_questions(questions)
    imgs = _resolve_images(images)

    attempts = _candidates(model, engine, cfg)
    errors: list[str] = []
    for eng, name, info in attempts:
        if imgs:
            if info.vision is False:
                errors.append(f"{eng}:{name}: no image support")
                continue
            if info.convention is None:
                errors.append(
                    f"{eng}:{name}: no documented image convention "
                    f"(vision-capable: {', '.join(vision_capable_models())})"
                )
                continue
            if info.max_images is not None and len(imgs) > info.max_images:
                errors.append(f"{eng}:{name}: max {info.max_images} images (got {len(imgs)})")
                continue
        try:
            payload = _build_payload(name, state, questions, imgs, info)
            url, headers = _resolve_url(eng, cfg)
            return _call_systemone(url, headers, payload, cfg.timeout)
        except EngineError as exc:
            errors.append(f"{eng}:{name}: {exc}")
    raise EngineError("all engine attempts failed — " + "; ".join(errors))