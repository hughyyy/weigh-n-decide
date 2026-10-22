# AGENTS.md — weigh-n-decide

## Project

MCP server giving agents decisions: executing decision models
(Jev/SystemOne-compatible endpoints — Clef, decider family, Mercury Decide,
GPT-6 Luna Decisions), vision decision models (the same questions API over
images), and literal condition evaluation (local, deterministic). Fifth in the
collection (look-n-see, scrape-n-search, store-n-recall, listen-n-speak).
**Status: v1 implemented** — `BRIEF.md` (requirements), `IMPLEMENTATION.md`
(design), `README.md` (user-facing surface), `CONTEXT.md` (vocabulary). The
remaining open questions at the bottom of README gate only what is not yet
built; resolve them with the user before building more.

## Conventions

- Use `CONTEXT.md` terms exactly: decision model, state, question, option,
  answer, condition, fact, trace, engine, model, input triple,
  vision-capable. They are not synonyms for anything else.
- SystemOne fidelity: the questions API (`state`; `questions` typed `noul` /
  `choice` / `score`; `instructions`; `criteria`; images) is mirrored as
  defined by the Clef and decider model cards — validate inputs, pass
  through, normalize only the response. Never invent request or response
  fields the cards do not define.
- Python 3.10+. Collection parity: setuptools src layout, package
  `weigh_n_decide`, console script `weigh-n-decide` (stdio transport), module
  `weigh_n_decide.server:main`. No second toolchain (Node/Cargo/…) without a
  stated reason the stdlib-plus-mcp baseline cannot serve.
- Dependency baseline: `mcp>=2` + `httpx>=0.27` (parity with look-n-see's
  `pyproject.toml`). Everything else stdlib. Tests use stdlib `unittest`,
  zero dev deps.
- Determinism split: `evaluate` is pure local compute — no network, no
  randomness, stable ordering. `decide`/`vision_decide` are the only
  networked tools, stateless HTTP per call.
- Stateless tools: state, questions, and facts are passed per call; nothing
  persisted server-side.
- Every tool returns structured JSON the calling model can act on
  (probabilities included — thresholding is the caller's job). Missing
  inputs, unreachable endpoints, and non-vision-capable models degrade to
  clear errors, never tracebacks.
- Engine pluggability follows look-n-see's pattern: engine/model param per
  call, env-configured default, fallback chain.
- Config via environment variables (`WEIGH_N_DECIDE_*`) with sane defaults;
  secrets from env only, never hardcoded or committed.
- OMP's default MCP request timeout is 30 s — `evaluate` must stay
  sub-millisecond; `decide`/`vision_decide` need a raised registration
  timeout (README shows 120 s).
- Tool surface stays minimal (`decide`, `evaluate`, `vision_decide`). A new
  tool needs a README row and a reason it cannot be a param; a new engine,
  question type, or env var needs a README update in the same change.

## Workflow

- The design in `README.md` is the draft spec. Open questions are resolved
  with the user first, one at a time, each framed with its problem context,
  options, and recommendation — never batched.
- When implementation starts, create `BRIEF.md` (requirements) and
  `IMPLEMENTATION.md` (design + release plan) like look-n-see, and keep them
  in sync with decisions made during implementation.
- Docs precede code: a change to the tool surface, grammar, or engine set
  lands in README first, then implementation, in the same change.
- One commit per logical change; conventional commits, imperative subject.
  Docs-only changes say so.

## Verification

- `python -m compileall` / import smoke test after changes.
- `python -m unittest discover -s tests -q` — the offline suite: the
  condition evaluator is fully unit-testable; engine request/response mapping
  is table-driven against a fake `/v1/systemone` (stdlib `http.server`,
  in-process). No network in tests.
- Server must start and register tools with no keys and no reachable endpoint
  — `evaluate` fully functional; `decide`/`vision_decide` error clearly on
  use, no network at startup.
- Live calls are manual smoke (a real MCP call through the harness against
  SGLang/decider.serve or OpenRouter), not part of the suite.