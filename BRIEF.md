# BRIEF — weigh-n-decide

Requirements for the weigh-n-decide MCP server. Design lives in
`IMPLEMENTATION.md`; the tool surface and config table live in `README.md`.

## Problem

An agent harness has no plumbing for decision models — the new model class
(Clef, decider, Mercury Decide, GPT-6 Luna Decisions) that reads a state and
answers typed questions with calibrated probabilities over the shared
Jev/SystemOne API. Each family ships its own server or host; an agent would
have to know every endpoint, format, and key. It also re-reasons pure boolean
logic over facts in its own context, which is unreliable and expensive for
compound conditions.

## Requirements

1. **One stable MCP surface** (`decide`, `vision_decide`, `evaluate`) in front
   of many decision-model backends.
2. **SystemOne fidelity**: the questions API (`state`; `questions` typed
   `noul`/`choice`/`score`; `instructions`; `criteria`; images) is mirrored as
   the model cards define it. Validate inputs strictly; never invent request
   or response fields.
3. **Engine pluggability** (family pattern): engine/model per call,
   env-configured default, fallback chain; missing endpoint or key degrades
   to a clear error, never a traceback, never a startup failure.
4. **Vision decision models**: images enter via the input triple (path / data
   URI / URL); non-vision-capable models are rejected with an error naming
   alternatives; per-backend image conventions handled by the engine.
5. **Deterministic conditions**: literal boolean/comparison grammar
   (booleans + comparisons only — arithmetic and functions are out of v1
   scope, decided with the user), recursive-descent parser, no `eval()`,
   per-node trace, missing fact path is an error.
6. **Stateless tools**: state, questions, facts passed per call; nothing
   persisted server-side; `evaluate` never touches the network.
7. **Structured output**: every tool returns JSON the calling model can act
   on (full probability distributions included — thresholding is the
   caller's job).

## Acceptance criteria

- Offline suite green (`python -m unittest discover -s tests -q`): condition
  grammar fully covered; engine request/response mapping covered against a
  fake `/v1/systemone` (stdlib `http.server`, in-process); no network in
  tests.
- Server starts and registers all three tools with no endpoint and no key;
  remote paths error clearly at call time (verified by
  `scripts/stdio_smoke.py` over real MCP stdio).
- A live `decide` against SGLang Clef / decider.serve and an OpenRouter
  decision model returns normalized answers (manual smoke, not in suite).

## Non-goals (v1)

- No local weight loading (torch/GPU); serving backends is the operator's job.
- No prompting adapter for plain instruct models (Muse2-230M etc.).
- No weighted-criteria scoring engine — rubrics are `score` questions.
- No model storage or sessions.