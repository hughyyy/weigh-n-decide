"""Model registry: known decision models, their engine, and image support."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """What the server knows about one decision model.

    `vision`: True = reads images, False = text-only, None = unknown (the
    card does not say). `convention`: how images enter the SystemOne request
    — "assets" (decider: state references keyed assets) or "images" (clef:
    an image list). None = unknown / not applicable.
    """

    engine: str
    vision: bool | None = None
    max_images: int | None = None
    convention: str | None = None


# Engine "selfhosted" = any POST /v1/systemone base URL (SGLang Clef,
# decider.serve). Engine "openrouter" = OpenRouter-hosted System One models.
MODELS: dict[str, ModelInfo] = {
    # Cloudflare Clef (SGLang serves /v1/systemone; images and video in card)
    "clef": ModelInfo(engine="selfhosted", vision=True, convention="images"),
    "clef-flash": ModelInfo(engine="selfhosted", vision=True, convention="images"),
    # Mapika decider family (decider.serve; assets referenced from the state,
    # PNG/JPEG, DECIDER_MAX_IMAGES default 4)
    "decider-2b": ModelInfo(engine="selfhosted", vision=True, max_images=4, convention="assets"),
    "decider-4b": ModelInfo(engine="selfhosted", vision=True, max_images=4, convention="assets"),
    "decider-12b": ModelInfo(engine="selfhosted", vision=False),
    "decider-12b-vision": ModelInfo(engine="selfhosted", vision=True, max_images=4, convention="assets"),
    "decider-31b": ModelInfo(engine="selfhosted", vision=True, max_images=4, convention="assets"),
    "decider-35b-a3b": ModelInfo(engine="selfhosted", vision=False),
    # OpenRouter-hosted decision models (vision capability not documented)
    "inception/mercury-decide": ModelInfo(engine="openrouter"),
    "inception/mercury-decide:free": ModelInfo(engine="openrouter"),
    "openai/gpt-6-luna-decisions": ModelInfo(engine="openrouter"),
}

DEFAULT_MODEL = "clef"

_VISION_CAPABLE = sorted(
    name for name, info in MODELS.items() if info.vision
)


def resolve_model(
    model: str | None, engine: str | None, default_model: str, default_engine: str
) -> tuple[str, str, ModelInfo]:
    """Resolve a model/engine pair to (engine, model_id, info).

    A model containing "/" that is not in the registry is treated as an
    OpenRouter id. Bare unknown names are an error naming the known models.
    """
    name = (model or default_model).strip()
    if not name:
        raise ValueError("model name is empty")
    info = MODELS.get(name)
    if info is None:
        if "/" in name:
            info = ModelInfo(engine="openrouter")
        else:
            known = ", ".join(sorted(MODELS))
            raise ValueError(
                f"unknown model {name!r}; known models: {known} "
                f"(or use a provider-qualified id like 'vendor/model')"
            )
    engine_name = (engine or info.engine or default_engine).strip()
    return engine_name, name, info


def vision_capable_models() -> list[str]:
    """Names of models documented to read images."""
    return list(_VISION_CAPABLE)