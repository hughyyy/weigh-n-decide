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
  types, wrong criteria shapes, unsupported image formats, duplicate refs,
  and malformed image entries are input errors raised before any dispatch;
  they are not fallback-worthy. Endpoint/key/response errors walk the
  fallback chain and aggregate every attempt into the final error. A
  malformed fallback entry is aggregated too, but never prevents a live
  primary from succeeding.
- **Model registry lives in code** (`registry.py`), not a config file: one
  place, reviewable, versioned with the code. `model` resolution: an explicit
  per-call `engine` wins, then an explicitly configured
  `WEIGH_N_DECIDE_ENGINE`, then the model's registry engine; unknown names
  containing `/` default to OpenRouter; bare unknown names are errors listing
  known models. Env `WEIGH_N_DECIDE_MODEL` sets the default.
- **Image conventions are per model** (`registry.convention`): "assets" for
  the decider family (`{"type": "image", "format", "data_base64"}` keyed by
  ref, refs default `image_0…`), "images" for clef (data-URI list — inferred;
  the clef card documents `images` on the SystemOne request without a wire
  example). Unknown convention + images → clear error naming vision-capable
  models. Capability, convention, and `max_images` are checked before any
  image is loaded, so an unsupported model never triggers a file read or URL
  fetch. Image entries reject unknown fields and blank refs; data URIs need
  canonical base64; schemes are matched case-insensitively; non-http(s)
  redirects are refused. Reads are bounded (25 MiB per image, 100 MiB per
  call), paths must be regular files, and `Content-Length` over the limit is
  rejected before the body is read. URL targets are restricted to globally
  routable addresses by default (`WEIGH_N_DECIDE_IMAGE_URL_POLICY=public`);
  `any` opts into loopback/private targets.
- **Conditions AST** carries a `start` offset per node so every trace node
  can quote its source text (parenthesized groups quote their parentheses,
  and parentheses also wrap operands such as `(a.total) > 100`). Trace:
  `{"expr", "value", "op", children…}` with `skipped` listing short-circuited
  sub-expressions. Input is bounded at 20,000 characters, 5,000 tokens, and
  100 explicit nesting levels (parentheses, `not`, list literals) so
  pathological input is a `ConditionError`, not a stack overflow. Number
  literals follow JSON (including exponents); escapes include `\/`, `\b`,
  `\f`, and `\uXXXX` with surrogate pairs combining; raw newlines in strings
  are rejected. JSON equality is strict about booleans (`true != 1`) —
  Python's `bool`-is-`int` trap is guarded.
- **Determinism split**: `evaluate` is pure local compute; `decide`/
  `vision_decide` are the only networked tools. Responses are normalized
  (`model`/`answers`/`usage`) but individual answer fields pass through
  unchanged until per-backend card parity is confirmed (README open question).

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

- Offline: `python -m unittest discover -s tests -q` — 379 tests, about 8 s.
  Coverage is 100% line/branch for `conditions`, `config`, `engines`,
  `images`, and `registry`; the package total is 99% line and every branch is
  covered except the `if __name__ == "__main__"` guard branch in `server.py`,
  which the stdio subprocess test executes. The suite covers the grammar
  (truth-table sweeps, a type-pair × operator matrix against a reference
  interpreter, seeded differential sweeps, generated paths, trace
  self-consistency, junk-input fuzzing, JSON number/escape parity, the 20k
  character / 5k token / 100-level bounds including list literals), traces
  (verbatim source text and skipped parts), surrogate pairs and lone
  surrogates, determinism, config defaults and invalid env values, registry
  precedence, engine validation (state types including non-finite values,
  question shapes, image specs), both image conventions, image capability
  checked before loading, per-image and total size caps, regular-file
  enforcement, bounded/truncated URL reads, URL policy (`public` vs `any`,
  DNS and port failures), redirect guards, fallback chains (engine-only,
  model-only, colon-suffixed ids, bad entries non-fatal, blank-engine dedup,
  unknown-engine aggregation, no fallback after success), candidate dedup,
  endpoint path construction, response normalization, auth headers,
  invalid-timeout/read-timeout degradation, overall deadlines before and
  during image loading, timeout clamping, a persistent httpx client,
  tool behavior via `mcp.call_tool`, and a real stdio end-to-end session over
  JSON-RPC pipes for both entry points.
- Coverage is statement/branch evidence, not behavioral proof: the
  regressions fixed in this change (`WEIGH_N_DECIDE_ENGINE` being inert,
  malformed fallbacks aborting the primary, `\u` escape validation, list
  literal recursion, doubled endpoint paths, image loading before capability
  checks) all lived behind fully covered lines. Each now has a targeted test.
- Suite strength (ad-hoc, not part of the suite): an earlier mutation campaign
  over `conditions`, `config`, `registry`, and engine validation produced 178
  syntactically valid mutants; 176 were killed. The two survivors were
  semantically equivalent rewrites of the redundant `_json_equal` type guard.
  Re-run the campaign after grammar or validation changes.
- Smoke: `scripts/stdio_smoke.py` — real MCP stdio: initialize, tools/list,
  evaluate roundtrip with traces, clear error paths.
- Live: manual — point `WEIGH_N_DECIDE_ENDPOINT` at SGLang Clef or
  `decider.serve`, or set `OPENROUTER_API_KEY` for the OpenRouter slate.