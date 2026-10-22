"""MCP server exposing decide / vision_decide / evaluate."""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from . import conditions, engines
from .config import Config, load_config



def create_server(cfg: Config | None = None) -> MCPServer:
    """Build the MCP server. All three tools are always registered; remote
    engines fail with clear errors at call time, never at startup."""
    cfg = cfg or load_config()
    mcp = MCPServer(name="weigh-n-decide")

    @mcp.tool()
    def decide(
        state: str | dict | list,
        questions: dict,
        model: str | None = None,
        engine: str | None = None,
    ) -> str:
        """Execute a decision model (Jev/SystemOne API) over a state and typed questions.

        Args:
            state: The situation to decide on — a string or JSON value.
            questions: Map of question id -> typed question; answers come back
                with calibrated probabilities per option.
            model: Optional decision model id (clef, decider-2b,
                inception/mercury-decide, ...). Defaults to WEIGH_N_DECIDE_MODEL.
            engine: Optional engine: `selfhosted` (WEIGH_N_DECIDE_ENDPOINT) or
                `openrouter` (OPENROUTER_API_KEY).
        """
        return _decide(state, questions, model, engine, cfg)

    @mcp.tool()
    def vision_decide(
        images: list,
        state: str | dict | list,
        questions: dict,
        model: str | None = None,
        engine: str | None = None,
    ) -> str:
        """Execute a vision-capable decision model: the questions API with images in the state.

        Args:
            images: Images as file path, base64 data URI, or http(s) URL —
                bare strings or {"image": "...", "ref": "name"} objects. The
                state references assets by ref (default image_0, image_1, ...).
                Requires a vision-capable model (clef, clef-flash, decider-2b,
                decider-4b, decider-12b-vision, decider-31b).
            state: The situation to decide on; reference images by ref.
            questions: Map of question id -> typed question (see decide).
            model: Optional decision model id. Defaults to WEIGH_N_DECIDE_MODEL.
            engine: Optional engine: `selfhosted` or `openrouter`.
        """
        return _decide(state, questions, model, engine, cfg, images=images)

    @mcp.tool()
    def evaluate(conditions: str | list, facts: dict | None = None) -> str:
        """Evaluate literal boolean conditions against facts — deterministic, no model.

        Args:
            conditions: One expression or a list of expressions, e.g.
                'a.status == "paid" and (a.total > 100 or a.trial) and not a.vip'.
                Operators: and/or/not, ==/!=/</<=/>/>=/in/not in/contains,
                parentheses; operands are JSON literals and dotted/indexed fact
                paths (a.total, items[0].price, user.roles).
            facts: JSON object the fact paths read from. A missing path is an
                error, not a silent false.
        """
        return _evaluate(conditions, facts)

    return mcp


def _decide(
    state: object,
    questions: object,
    model: str | None,
    engine: str | None,
    cfg: Config,
    images: object = None,
) -> str:
    try:
        result = engines.run_decision(
            state, questions, images, model=model, engine=engine, cfg=cfg
        )
        return json.dumps(result)
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured tool result
        return json.dumps({"error": str(exc)})


def _evaluate(conditions_arg: str | list, facts: dict | None) -> str:
    try:
        return json.dumps(conditions.evaluate_conditions(conditions_arg, facts))
    except Exception as exc:  # noqa: BLE001 - surfaced as a structured tool result
        return json.dumps({"error": str(exc)})


def main() -> None:
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()