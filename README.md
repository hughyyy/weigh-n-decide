# weigh-n-decide

A lightweight [MCP](https://modelcontextprotocol.io) server that gives agents
**decision machinery**: **decide** executes decision models — weighted criteria
over alternatives, deterministic ranking — **evaluate** judges literal
conditions against facts with no model in the loop, and **vision_decide**
scores alternatives from images with a vision model before the same weighted
kernel ranks them. Fifth in the collection: [look-n-see](../look-n-see)
(vision/OCR), [scrape-n-search](../scrape-n-search) (web),
[store-n-recall](../store-n-recall) (memory), [listen-n-speak](../listen-n-speak)
(speech), weigh-n-decide (decisions).

> **Status: planned** — docs-first: this README is the draft spec, and the open
> design questions at the bottom gate the final surface. Implementation starts
> after they are resolved; then `BRIEF.md` (requirements) and
> `IMPLEMENTATION.md` (design + release plan) are added, like look-n-see.

## The idea

Three gaps this closes for an agent:

- **Literal logic is where models are least reliable.** A long compound
  condition over many facts — `a.status == "paid" and (a.total > 100 or a.trial)
  and not a.vip` — is exactly what a model fumbles. A deterministic evaluator
  returns the right boolean plus a **trace** of every sub-expression: auditable,
  reproducible, identical on re-run. No tokens are spent re-reasoning logic that
  is pure math.
- **Weighted decision models make tradeoffs explicit data.** Criteria, weights,
  and per-alternative scores live in a model that can be read, diffed, and
  re-run; the ranking comes from arithmetic, not vibes. Changing a weight is a
  one-line edit with an immediately comparable result.
- **Vision decision models connect evidence to scoring.** A VLM looks at each
  alternative's image and scores a rubric; the weighted kernel turns those
  scores into a ranking. look-n-see can describe an image; weigh-n-decide turns
  image evidence into a decision.

The server is **stateless** — models and facts are passed in with each call and
nothing is stored. Deterministic tools (`decide`, `evaluate`) never touch the
network; `vision_decide` is the only networked call.

## Tool surface (draft)

| Tool | Params | Returns |
| --- | --- | --- |
| `decide` | `model` (required: criteria + weights + alternatives + scores), `format`? (default `weighted`) | ranking: per-alternative weighted total, rank, per-criterion contributions, tie flags, gap warnings (unscored criteria) |
| `evaluate` | `conditions` (required: one expression or list), `facts`? (JSON object) | per-condition boolean `result` + `trace` of every sub-expression; optional `all`/`any` aggregate |
| `vision_decide` | `criteria` (required: id, weight, question), `alternatives` (required: id + image), `engine`?, `model`? | per-alternative per-criterion scores + one-line rationale, then the `decide` ranking |

Input and output discipline (house rules): images use the same input triple as
look-n-see (file path / base64 `data:` URI / `http(s)://` URL); every tool
returns structured JSON the calling model can act on, never prose-only.

## Decision models (draft)

Canonical `weighted` format — the only one committed for v1:

```json
{
  "criteria": [
    {"id": "cost",  "weight": 0.4, "direction": "minimize"},
    {"id": "speed", "weight": 0.6}
  ],
  "alternatives": [
    {"id": "cli",   "scores": {"cost": 9, "speed": 8}},
    {"id": "cloud", "scores": {"cost": 6, "speed": 9}}
  ]
}
```

- Scores are 0–10 (one decimal); `direction` defaults to `maximize`;
  `minimize` inverts the score before summing (a low cost score is *good*).
- Weights need not sum to 1; the kernel normalizes internally and reports the
  share each criterion contributed to each alternative's total.
- Missing scores are **gaps**: reported per alternative;
  `require_complete=true` turns gaps into an error instead.
- Ties are detected and reported, never broken silently.

`jev` is the other format named for `format=`; its definition is an open
question gating this surface (see below). The engine behind `decide` is
format-agnostic: a loader turns any format into the canonical model, and one
arithmetic kernel ranks it.

## Literal conditions (draft)

A small whitelisted grammar — recursive-descent parser, no `eval()`, nothing
but literals, fact paths, and these operators:

- Boolean: `and`, `or`, `not`, parentheses; short-circuit evaluation (the trace
  records where it stopped)
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`, `in`, `not in`, `contains`
- Operands: JSON literals (`5`, `"paid"`, `true`, `null`, `[1, 2]`), and
  **fact paths** — dotted/indexed references into `facts` (`a.total`,
  `items[0].price`, `user.roles`)

Facts are a JSON object passed per call. A missing fact path is an error, not a
silent `false`. Arithmetic and functions are open questions (see below); the
v1 grammar is comparisons over literals only.

## Vision decision models (draft)

`vision_decide` = rubric + evidence → scores → ranking:

1. Each criterion carries a `question` (the scoring prompt) — "how well does
   this UI handle a first-run user?", answered 0–10.
2. Each alternative carries one image via the input triple.
3. One VLM call per alternative scores **all** criteria for that image (rubric
   in the prompt, score + rationale per criterion) — N calls for N
   alternatives, issued in parallel.
4. Scores feed the same weighted kernel as `decide`; output adds per-criterion
   rationale next to each score.

Engines follow look-n-see's pattern: OpenRouter / HuggingFace Inference, engine
param per call, env-configured default, fallback chain, clear error without
keys.

## Configuration (draft)

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEIGH_N_DECIDE_VISION_ENGINE` | `openrouter` | default vision engine |
| `WEIGH_N_DECIDE_VISION_MODEL` | engine-dependent | default VLM |
| `WEIGH_N_DECIDE_VISION_FALLBACKS` | — | fallback chain tried after the default fails |
| `OPENROUTER_API_KEY` / `HF_TOKEN` | — | secrets, env only (same as look-n-see) |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/weigh-n-decide   # stdio MCP server
```

### Register with Oh My Pi

`decide`/`evaluate` run fine at default timeouts (sub-millisecond local
compute); `vision_decide` needs the raised timeout look-n-see uses:

```json
{
  "mcpServers": {
    "weigh-n-decide": {
      "type": "stdio",
      "command": "/absolute/path/to/weigh-n-decide/.venv/bin/weigh-n-decide",
      "timeout": 120000
    }
  }
}
```

## Non-goals

- No model storage, no saved sessions — stateless; persistent memory is
  store-n-recall's job.
- No LLM in `decide`/`evaluate` — deterministic tools stay deterministic; the
  only model call is `vision_decide`.
- No probabilistic/Bayesian/AHP variants in v1 — one kernel, executed well.
- No visualizations or UI — rankings and verdicts are JSON.

## Open design questions (gating the spec)

1. **Jev format** — what is the `jev` decision-model format (spec, or where it
   is defined)? Gates `format=` beyond the canonical `weighted`.
2. **Vision semantics** — is VLM-as-rubric-scorer the intended reading of
   "vision decision models", or pairwise image comparison (VLM picks a winner
   per pair; ranking from win counts)?
3. **Condition grammar scope** — arithmetic inside comparisons (`+ - * / %`)?
   Functions (`len`, `count`)? The v1 draft is comparisons over literals only.
4. **Score scale policy** — fixed 0–10 with one decimal, or accept any numeric
   scale and normalize? Interaction with `direction`.
5. **vision_decide budget** — one call per alternative (batched rubric) vs one
   per criterion; timeout headroom for N alternatives.