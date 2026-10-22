# AGENTS.md — weigh-n-decide

## Project

MCP server giving agents decision machinery: executing decision models
(weighted criteria over alternatives), vision decision models (VLM-scored
rubrics over images), and literal condition evaluation. Fifth in the collection
(look-n-see, scrape-n-search, store-n-recall, listen-n-speak). **Status:
docs-first, implementation not started** — the design lives in `README.md`
(draft tool surface, formats, grammar, config) and `CONTEXT.md` (vocabulary).
The open design questions at the bottom of README gate the spec; resolve them
with the user before implementing.

## Conventions

- Use `CONTEXT.md` terms exactly: decision model, alternative, criterion,
  score, ranking, condition, fact, trace, kernel, engine, input triple,
  question. They are not synonyms for anything else.
- Python 3.10+. Collection parity: setuptools src layout, package
  `weigh_n_decide`, console script `weigh-n-decide` (stdio transport), module
  `weigh_n_decide.server:main`. No second toolchain (Node/Cargo/…) without a
  stated reason the stdlib-plus-mcp baseline cannot serve.
- Dependency baseline: `mcp>=2` + `httpx>=0.27` (parity with look-n-see's
  `pyproject.toml`). Everything else stdlib; engines behind extras. Tests use
  stdlib `unittest`, zero dev deps.
- Determinism: `decide` and `evaluate` are pure local compute — no network, no
  randomness, stable output ordering; same input → same ranking/verdict.
  `vision_decide` is the only networked tool.
- Stateless tools: models and facts are passed per call; nothing persisted
  server-side.
- Every tool returns structured output the calling model can act on (JSON
  where it matters, not prose-only). Missing inputs, keys, or fact paths
  degrade to clear errors, never tracebacks.
- Engine pluggability follows look-n-see's pattern: engine param per call,
  env-configured default, fallback chain.
- Config via environment variables (`WEIGH_N_DECIDE_*`) with sane defaults;
  secrets from env only, never hardcoded or committed.
- OMP's default MCP request timeout is 30 s — `decide`/`evaluate` must stay
  sub-second; `vision_decide` needs a raised registration timeout (README
  shows 120 s).
- Tool surface stays minimal (`decide`, `evaluate`, `vision_decide`). A new
  tool needs a README row and a reason it cannot be a param; a new format,
  operator, or env var needs a README update in the same change.

## Workflow

- The design in `README.md` is the draft spec. Open questions are resolved
  with the user first, one at a time, each framed with its problem context,
  options, and recommendation — never batched.
- When implementation starts, create `BRIEF.md` (requirements) and
  `IMPLEMENTATION.md` (design + release plan) like look-n-see, and keep them
  in sync with decisions made during implementation.
- Docs precede code: a change to the tool surface, grammar, or formats lands
  in README first, then implementation, in the same change.
- One commit per logical change; conventional commits, imperative subject.
  Docs-only changes say so.

## Verification

- `python -m compileall` / import smoke test after changes.
- `python -m unittest discover -s tests -q` — the offline suite. The kernel
  and condition evaluator are fully unit-testable; determinism (same input →
  same output) is a test target.
- Server must start and register tools with no keys present
  (`decide`/`evaluate` fully functional offline; `vision_decide` errors
  clearly on use, no network at startup).
- Live VLM calls are manual smoke (a real MCP call through the harness), not
  part of the suite.