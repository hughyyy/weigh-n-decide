# weigh-n-decide

A lightweight MCP server that gives agents decision machinery: executing weighted decision models over alternatives, scoring alternatives from images with a vision model, and evaluating literal conditions against facts.

## Language

### Artifacts

**Decision model**:
The structured input to `decide`: criteria with weights and directions, alternatives with per-criterion scores. Format-checked before execution; the canonical format is `weighted`.
_Avoid_: matrix, rubric (the rubric is the vision scoring prompt, not the model), model (ambiguous — say decision model or VLM)

**Alternative**:
One candidate being ranked. Has an id, optional human label, and scores — or, in `vision_decide`, an image that gets scored.
_Avoid_: option, candidate, choice, item

**Criterion**:
One weighted dimension of a decision: id, weight, direction (`maximize` default, `minimize` inverts before summing).
_Avoid_: factor, dimension, aspect, metric

**Score**:
The numeric grade (0–10) of one alternative on one criterion. Passed explicitly in `decide`; produced by the vision engine in `vision_decide`.
_Avoid_: rating, value, weight (a weight belongs to a criterion, not an alternative)

**Ranking**:
The output of decision-model execution: alternatives ordered by weighted total, with per-criterion contributions, shares, gaps, and ties.
_Avoid_: result, decision (that is the human's call), winner

**Condition**:
One literal boolean expression evaluated against facts, producing a result plus a trace. Evaluated deterministically — no model in the loop.
_Avoid_: rule (a rule set would be a grouping of conditions, not v1 scope), expression (fine internally, not in docs), predicate

**Fact**:
The JSON object a condition reads through fact paths. A missing path is an error, never a silent false.
_Avoid_: data, context, state, variables

**Trace**:
The per-node record of one condition's evaluation: sub-expression values, short-circuit points, resolved fact paths.
_Avoid_: log, audit, proof

### Machinery

**Kernel**:
The one arithmetic module that turns a canonical decision model into a ranking. All formats load into it; nothing else computes scores.
_Avoid_: engine (reserved for vision engines), calculator, evaluator

**Engine**:
A swappable VLM implementation behind `vision_decide` — OpenRouter or HuggingFace Inference. Selected per call, env-configured default, fallback chain. Same pattern as look-n-see.
_Avoid_: backend, provider, model (an engine has a model; it is not one)

**Input triple**:
The three accepted image forms in `vision_decide`: file path, base64 `data:` URI, `http(s)://` URL. Same normalization as look-n-see.
_Avoid_: image source, uri (only one of the three)

**Question**:
The per-criterion scoring prompt a vision engine answers with a 0–10 score and a one-line rationale.
_Avoid_: prompt (too broad), criterion (the question belongs to one)