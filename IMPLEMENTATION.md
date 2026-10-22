# IMPLEMENTATION — weigh-n-decide

Design and release plan. Requirements in `BRIEF.md`; the user-facing surface
in `README.md`.

## Module map

```
src/weigh_n_decide/
├── server.py     MCPServer: registers decide / vision_decide / evaluate;
│                 errors surfaced as {"error": ...} JSON, never tracebacks
├── engines.py    input validation (strict SystemOne shapes), image
│                 resolution, request building, HTTP dispatch, fallback chain
├── registry.py   model table: engine, vision capability, max images,
│                 image convention
├── config.py     env-driven Config dataclass (WEIGH_N_DECIDE_*)
├── conditions.py literal condition grammar: tokenizer, recursive-descent
│                 parser, evaluator with per-node traces
└── images.py     input triple (path / data URI / URL) → (media_type, bytes)
```

## Design decisions

- **One wire format, thin adapters.** Every v1 backend speaks Jev/SystemOne
  (`POST /v1/systemone`), so `engines.py` is one dispatcher: `selfhosted`
  (base URL from `WEIGH_N_DECIDE_ENDPOINT`) and `openrouter` (full endpoint
  URL from `WEIGH_N_DECIDE_OPENROUTER_URL`, Bearer auth). Engines differ only
  in endpoint resolution and auth headers.
- **Validation is strict and up front.** Unknown question fields, unknown
  types, wrong criteria shapes, unsupported image formats, and duplicate refs
  are input errors raised before any dispatch; they are not fallback-worthy.
  Endpoint/key/response errors walk the fallback chain and aggregate every
  attempt into the final error.
- **Model registry lives in code** (`registry.py`), not a config file: one
  place, reviewable, versioned with the code. `model` resolution: registry
  hit → its engine; unknown name containing `/` → OpenRouter id; bare unknown
  name → error listing known models. Env `WEIGH_N_DECIDE_MODEL` sets the
  default.
- **Image conventions are per model** (`registry.convention`): "assets" for
  the decider family (`{"type": "image", "format", "data_base64"}` keyed by
  ref, refs default `image_0…`), "images" for clef (data-URI list — inferred;
  the clef card documents `images` on the SystemOne request without a wire
  example). Unknown convention + images → clear error naming vision-capable
  models.
- **Conditions AST** carries a `start` offset per node so every trace node
  can quote its source text. Trace: `{"expr", "value", "op", children…}` with
  `skipped` listing short-circuited sub-expressions. JSON equality is strict
  about booleans (`true != 1`) — Python's `bool`-is-`int` trap is guarded.
- **Determinism split**: `evaluate` is pure local compute; `decide`/
  `vision_decide` are the only networked tools.

## Resolved design questions (from the README open list)

- Vision semantics: vision decision models = decision models that read images
  as part of the state (per the clef/decider cards) — `vision_decide` is
  `decide` + images.
- Condition grammar scope: booleans + comparisons only (user decision);
  arithmetic/functions can join later without changing the surface.
- Image conventions: per-model `convention` in the registry (see above).
- Model registry: static table in `registry.py` (see above).

## Release plan

- **v1 (this)**: three tools, two engines, registry, offline suite, stdio
  smoke. Remaining open items: OpenRouter System One route (default
  `…/api/v1/systemone` is a working guess, env-overridable), Jev endpoint
  reachability, `score` answer field parity across backends.
- **Next candidates** (only with a README row first): arithmetic in
  conditions, more engines, prompting adapter for instruct models.

## Verification

- Offline: `python -m unittest discover -s tests -q` — 76 tests covering the
  grammar, traces, engine validation, both image conventions, fallback
  chains, auth headers, and tool behavior via `mcp.call_tool` against the
  fake endpoint.
- Smoke: `scripts/stdio_smoke.py` — real MCP stdio: initialize, tools/list,
  evaluate roundtrip with traces, clear error paths.
- Live: manual — point `WEIGH_N_DECIDE_ENDPOINT` at SGLang Clef or
  `decider.serve`, or set `OPENROUTER_API_KEY` for the OpenRouter slate.