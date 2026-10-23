# weigh-n-decide

A lightweight [MCP](https://modelcontextprotocol.io) server that gives agents
**decisions**: **decide** executes decision models — small, fast models that
read a state and answer typed questions with calibrated probabilities
(Clef, decider, Mercury Decide, GPT-6 Luna Decisions, …, all sharing the
Jev/SystemOne API) — **vision_decide** runs the same questions API with images
in the state, and **evaluate** judges literal conditions against facts
locally, with no model in the loop. Fifth in the collection:
[look-n-see](../look-n-see) (vision/OCR), [scrape-n-search](../scrape-n-search)
(web), [store-n-recall](../store-n-recall) (memory),
[listen-n-speak](../listen-n-speak) (speech), weigh-n-decide (decisions).

> **Status: v1 implemented** — tools, engines, registry, and condition
> evaluator are live (379-test offline suite; `scripts/stdio_smoke.py` drives
> the real stdio server). `BRIEF.md` holds requirements, `IMPLEMENTATION.md`
> the design. The open questions at the bottom gate only what remains.

## The idea

- **Decision models are a model class built to be called, not chatted with.**
  Instead of generating prose, they read a `state` (text, JSON, or images) and
  answer typed `questions` — true/false, named choices, ordered scores — with
  a probability for every option, in one forward pass. No output parsing,
  calibrated confidence, latencies of tens of milliseconds (clef-flash median
  38.8 ms on the Decision Index), costs of fractions of a cent ($0.02/M input
  for Mercury Decide).
- **An agent harness has none of this plumbing.** Each family (Cloudflare
  Clef, Mapika decider, Inception Mercury, OpenAI Luna) ships its own server
  or host; the wire format they share is the Jev/SystemOne API
  (`POST /v1/systemone`). weigh-n-decide is the adapter: one stable MCP
  surface in front of many decision-model backends, with per-call model
  selection, env-configured defaults, and a fallback chain — the same engine
  pattern as look-n-see's OCR dispatcher.
- **Literal conditions stay local.** Compound boolean logic over facts is pure
  math: a deterministic evaluator returns the right boolean plus a trace of
  every sub-expression — auditable, reproducible, free. No model is consulted.

## The questions API (Jev / SystemOne)

All v1 backends speak one wire format, so the MCP surface mirrors it
faithfully — the server validates and passes through, normalizing only the
response:

- `state` — string or JSON describing the situation to decide on.
- `questions` — map of question id → question:
  - `noul` — true/false; optional `criteria` descriptions for true and false.
  - `choice` — named options; `criteria` maps option id → description.
  - `score` — ordered legend (array of descriptions; index = score value).
  - `instructions` — optional per-question instructions; the question id is
    used when they are omitted.
- Images (vision models) — `images`/`assets` in the request; see
  `vision_decide`.

Response: `model`, `answers` keyed by question id, `usage`. An answer carries
`choice` + `confidence` + `probabilities` (choice), `score` + `confidence` +
`legend` + `probabilities` (score), or the probability of true (noul). The
probabilities are the point: every option's full distribution comes back, so
the caller can threshold, compare, or defer.

## Tool surface

| Tool | Params | Returns |
| --- | --- | --- |
| `decide` | `state` (required: string or JSON), `questions` (required: map of typed questions), `model`?, `engine`? | `answers` per question id (normalized), `model`, `usage` |
| `vision_decide` | same as `decide` plus `images` (required: list of input-triple entries, optional `ref` each) | same; images folded into the state per backend convention |
| `evaluate` | `conditions` (required: one expression or list), `facts`? (JSON object) | per-condition boolean `result` + `trace`; `all`/`any` aggregate |

House rules: images use the same input triple as look-n-see (file path /
base64 `data:` URI / `http(s)://` URL); every tool returns structured JSON,
never prose-only; `evaluate` never touches the network.

Image inputs are agent-supplied, so the server bounds them: 25 MiB per image,
100 MiB total per call, bounded reads (a large `Content-Length` is rejected
before the body is read), regular files only for paths, http(s)-only
redirects, and — by default — URL hosts must resolve to globally routable
addresses. Set `WEIGH_N_DECIDE_IMAGE_URL_POLICY=any` to allow loopback or
private targets explicitly.

## Backends

| Engine | Endpoint | Models | Needs |
| --- | --- | --- | --- |
| `selfhosted` | any `POST /v1/systemone` base URL — SGLang serving Clef, `decider.serve` | clef, clef-flash, decider-2b/4b/12b/12b-vision/31b/35b-a3b | local or reachable server; no key |
| `openrouter` | OpenRouter's System One endpoint | inception/mercury-decide (and `:free`), openai/gpt-6-luna-decisions | `OPENROUTER_API_KEY` |

Selection per call via `engine=`/`model=`, default via env, fallback chain
after failure — identical semantics to look-n-see's OCR dispatcher. A model's
registry engine is used unless `engine=` or `WEIGH_N_DECIDE_ENGINE` explicitly
overrides it; unknown model ids containing `/` default to `openrouter`. Bare
unknown model names are errors. A plain instruct model (Muse2-230M and
friends) is not SystemOne-compatible and is out of scope for v1.

## Vision decision models

`vision_decide` is `decide` with images in the state — the same typed
questions, answered about picture evidence. Vision-capable models: clef /
clef-flash (images and video), decider-2b v12+ (assets referenced from the
state, PNG/JPEG, max 4), decider-12b-vision, decider-31b. The tool accepts the
input triple per image with an optional `ref`; adapters map to backend
convention (clef `images` list; decider `assets` map the state references by
key). Routing a non-vision-capable model with images fails with a clear error
naming which models would work.

## Literal conditions

A small whitelisted grammar — recursive-descent parser, no `eval()`, nothing
but literals, fact paths, and these operators:

- Boolean: `and`, `or`, `not`, parentheses; short-circuit evaluation (the
  trace records where it stopped). Parentheses group whole expressions and
  can also wrap operands (`(a.total) > 100`, `a == (1)`).
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `contains`
- Operands: JSON literals (`5`, `1.5e3`, `"paid"`, `true`, `null`, `[1, 2]`)
  and **fact paths** — dotted/indexed references into `facts` (`a.total`,
  `items[0].price`, `user.roles`). JSON escapes (`\/`, `\b`, `\f`, `\uXXXX`)
  follow the standard; raw newlines inside strings are rejected.

Facts are a JSON object passed per call. A missing fact path is an error, not
a silent `false`. Expressions are capped at 20,000 characters / 5,000 tokens
and 100 explicit nesting levels (parentheses, `not`, list literals).
Arithmetic and functions are deliberately out of v1 scope (decided with the
user); the grammar can grow later without changing the surface.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEIGH_N_DECIDE_ENGINE` | model's own engine | default engine override (`selfhosted` or `openrouter`); unknown values surface as a call-time error |
| `WEIGH_N_DECIDE_ENDPOINT` | — | base URL for `selfhosted` (SGLang / decider.serve); a full `.../v1/systemone` URL is also accepted |
| `WEIGH_N_DECIDE_MODEL` | `clef` | default decision model |
| `WEIGH_N_DECIDE_FALLBACKS` | — | fallback engines/models tried after the default fails (`engine:model` or `model`, comma-separated; model ids containing `/` or `:` are supported) |
| `WEIGH_N_DECIDE_OPENROUTER_URL` | `https://openrouter.ai/api/v1/systemone` | full System One endpoint for the `openrouter` engine (route unverified — open question) |
| `WEIGH_N_DECIDE_TIMEOUT` | `60` | per-request HTTP timeout in seconds for SystemOne calls and image fetches; blank, non-numeric, non-finite, or non-positive values fall back to the default |
| `WEIGH_N_DECIDE_DEADLINE` | `120` | overall budget in seconds for one `decide`/`vision_decide` call, including image loading and every fallback attempt |
| `WEIGH_N_DECIDE_IMAGE_URL_POLICY` | `public` | `public` rejects loopback/link-local/private image URL targets; `any` allows them |
| `OPENROUTER_API_KEY` | — | `openrouter` engine only; env only, never committed |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/weigh-n-decide   # stdio MCP server
```

### Register with Oh My Pi

`evaluate` runs at default timeouts (pure local compute); `decide`/
`vision_decide` call remote models — register with the raised timeout
look-n-see uses:

```json
{
  "mcpServers": {
    "weigh-n-decide": {
      "type": "stdio",
      "command": "/absolute/path/to/weigh-n-decide/.venv/bin/weigh-n-decide",
      "env": {
        "OPENROUTER_API_KEY": "!/home/<user>/.local/bin/omp token openrouter"
      },
      "timeout": 120000
    }
  }
}
```

The `!` prefix makes OMP run the command at spawn time and inject its stdout
into the server's environment — the key comes from OMP's credential store
(`omp token openrouter`), never from a committed file. The server reads env
only, so this is the whole wiring. `selfhosted` needs no key, only
`WEIGH_N_DECIDE_ENDPOINT` (also injectable here). `HF_TOKEN` rides the same
path if an HF-hosted engine is ever added; v1's HF-hosted models (clef,
decider) are self-served via SGLang/decider.serve and need none.

## Non-goals

- No local weight loading (torch/GPU) — backends are HTTP endpoints; serving
  SGLang/decider.serve is the operator's job.
- No prompting adapter for plain instruct models in v1 — Muse2-230M etc. are
  not SystemOne models.
- No weighted-criteria scoring engine — rubrics are expressed as `score`
  questions; the decision models are the scorer.
- No model storage, no sessions — stateless; persistent memory is
  store-n-recall's job.
- No visualizations — answers, probabilities, and traces are JSON.

## Open questions (remaining)

1. **OpenRouter System One route** — the default `…/api/v1/systemone` is a
   working guess (`WEIGH_N_DECIDE_OPENROUTER_URL` overrides it); verify the
   real route and auth for mercury-decide / gpt-6-luna-decisions, plus
   response-shape parity with `/v1/systemone`.
2. **Jev itself** — where is a Jev endpoint reachable (self-hosted? a host?),
   or do Clef/decider/Mercury cover the v1 slate?
3. **Score answers** — confirm `score`/`legend`/`confidence` field parity
   across backends.

Resolved questions (vision semantics, grammar scope, image conventions,
registry format) are recorded in `IMPLEMENTATION.md`.