# weigh-n-decide

A lightweight MCP server that gives agents decisions: executing decision models (Jev/SystemOne-compatible endpoints) over a state and typed questions, running the same questions API with images in the state, and evaluating literal conditions against facts locally.

## Language

### Artifacts

**Decision model**:
A model that reads a state and answers typed questions with a probability for every option, in one pass — no prose generation. Reached as a Jev/SystemOne endpoint: clef, decider family, Mercury Decide, GPT-6 Luna Decisions.
_Avoid_: LLM, chat model, classifier (broader), judge, scorer

**State**:
The situation being decided on: a string or JSON value, optionally referencing images. Passed per call, never stored.
_Avoid_: input, context, prompt, content

**Question**:
One typed ask about the state — `noul` (true/false), `choice` (named options), `score` (ordered legend) — with optional `instructions`. The question id names the ask when instructions are omitted.
_Avoid_: task, query, criterion (criteria describe choice options; they are not the question)

**Option**:
One allowed answer of a question: an option id with a description (`choice`), a legend entry (`score`), or true/false (`noul`).
_Avoid_: choice (that is the answer's field name), alternative, label

**Answer**:
The normalized response for one question: `choice` + `confidence` + `probabilities`, `score` + `confidence` + `legend` + `probabilities`, or the probability of true (`noul`). The full distribution is the point — thresholding happens in the caller.
_Avoid_: result, verdict, decision (that is the human's call)

**Condition**:
One literal boolean expression evaluated against facts, producing a result plus a trace. Evaluated deterministically — no model in the loop.
_Avoid_: rule, expression (fine internally, not in docs), predicate

**Fact**:
The JSON object a condition reads through fact paths. A missing path is an error, never a silent false.
_Avoid_: data, context, state (reserved for decision models)

**Trace**:
The per-node record of one condition's evaluation: sub-expression values, short-circuit points, resolved fact paths.
_Avoid_: log, audit, proof

### Machinery

**Engine**:
A swappable adapter behind `decide`/`vision_decide` that turns MCP calls into `POST /v1/systemone` requests — `selfhosted` (SGLang Clef, decider.serve) or `openrouter`. Selected per call, env-configured default, fallback chain.
_Avoid_: backend, provider, model (an engine routes models; it is not one)

**Model**:
The specific decision model an engine routes to — `clef`, `decider-2b`, `inception/mercury-decide`. Selected per call, env-configured default.
_Avoid_: engine, endpoint

**Input triple**:
The three accepted image forms: file path, base64 `data:` URI, `http(s)://` URL. Same normalization as look-n-see; adapters convert to backend convention.
_Avoid_: image source, uri (only one of the three)

**Vision-capable**:
A decision model that reads images (sometimes video) as part of the state — clef/clef-flash, decider-2b v12+, decider-12b-vision, decider-31b. Routing a non-vision-capable model with images is an error, not a silent text-only call.
_Avoid_: multimodal (vague), image model, VLM (these models do not chat)