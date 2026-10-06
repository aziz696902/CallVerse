# CallVerse Quality Analyst V1

## Purpose and boundary

The Quality Analyst evaluates a completed text-based Customer Advisor
interaction from observable evidence. It asks whether the answer was accurate,
relevant, procedurally correct, compliant, likely to serve the customer, and
appropriate to the customer's sentiment. It also determines whether the
interaction needs supervisor review.

The Quality Analyst does not receive or store hidden chain-of-thought. Its input
contains the customer message, final advisor response, routing metadata,
resolution/escalation state, tool names, structured verified facts, explicit
customer-fact claims, policy citation IDs, action/approval events, and observable
procedure events. `QualityEvaluationInput.from_advisor_interaction` maps the
Phase 6 result and compact trace into this contract; structured facts and action
events can then be supplied by the tool/audit boundary.

## Processing design

```text
completed interaction evidence
  -> deterministic hard guardrails
  -> optional structured quality judge
  -> deterministic guardrail score caps and flag merge
  -> deterministic weighted overall score
  -> supervisor-review decision
```

The deterministic layer always runs first. The six nuanced scores exist only
when a validated structured judge result is available. Offline guardrails do
not pretend to produce AI scores.

## Delivery/e-commerce rubric

Every dimension uses a bounded 1–5 score:

- **1 — serious failure:** harmful, invented, unauthorized, or dismissive;
- **2 — significant problem:** a material gap a supervisor should address;
- **3 — acceptable baseline:** essential service was adequate;
- **4 — strong:** accurate, tailored, safe, and operationally useful;
- **5 — exceptional:** complete, precise, empathetic, and highly auditable.

The six explicit dimensions and expert-defined V1 weights are:

| Dimension | Weight |
|---|---:|
| factual accuracy | 0.25 |
| procedure adherence | 0.20 |
| compliance | 0.20 |
| relevance | 0.15 |
| customer satisfaction | 0.10 |
| sentiment handling | 0.10 |

The detailed definitions in `callverse/quality/rubric.py` are specific to
tracking, delivery, refunds, cancellation, address changes, damaged items,
payment boundaries, and escalation. These weights are provisional expert
choices, not learned parameters. The judge never supplies the final arithmetic
score: Python recomputes it deterministically from the six validated scores.

## Deterministic guardrails

V1 applies hard checks only where evidence is explicit:

- an executed refund or protected action without an approved decision produces
  a critical compliance flag and caps compliance/procedure scores;
- a customer-specific claim without a referenced business fact is flagged;
- a claim contradicting a referenced order fact is flagged and factual accuracy
  is capped;
- missing observable required procedure events produce `procedure_bypass`;
- a nonexistent or unverifiable order that is escalated without factual claims
  is recorded as a safe escalation, not penalized as a failed resolution;
- a refund draft or action stopped at the approval boundary is recorded as
  approval respected.

Supported operational flag codes include `unauthorized_refund`,
`invented_order_status`, `unsupported_customer_fact`,
`contradictory_customer_fact`, `missing_required_approval`,
`unsafe_account_change`, `procedure_bypass`, and
`unsafe_missing_order_handling`. High and critical severities are reserved for
meaningful operational risk. Hard flags and score caps cannot be removed by a
favorable judge response.

This is deliberately not a general natural-language fact checker. CallVerse
requires explicit `VerifiedFact` and `CustomerFactClaim` records for
deterministic factual comparison.

## Structured quality judge

`GroqStructuredQualityJudge` reuses HelpPilot's installed `ChatGroq` provider,
`GROQ_API_KEY`, and configured reviewer model. It does not introduce another
agent platform, LangGraph, provider abstraction, or vector database. Temperature
is zero and output is validated as `QualityJudgeOutput`.

The judge receives:

- the customer request and advisor response;
- classifier/routing and resolution metadata;
- verified tool facts and explicit claims;
- tool names and RAG citation IDs;
- approval/action and procedure evidence;
- the complete CallVerse rubric.

It returns six scores with concise justifications, meaningful quality/compliance
flags, a compact customer-sentiment label, and optional confidence. It does not
return an overall score.

If the key is missing, the provider fails, times out, or returns malformed
output, status becomes `unavailable`, nuanced scores and overall score remain
`None`, deterministic flags are preserved, and supervisor review is required.
The interaction is never marked good by default.

On 2026-10-06, no `GROQ_API_KEY` was available. The live quality judge and live
repeatability check were therefore not verified. This does not block the
offline architecture.

## Sentiment handling

The structured judge may classify customer sentiment as `positive`, `neutral`,
`frustrated`, or `angry`, and separately scores whether the advisor handled that
sentiment appropriately. The regression fixtures distinguish calm, specific
empathy from dismissive treatment of an angry customer.

The presentation proposed XLM-R sentiment as a possible component. V1 does not
download or claim an XLM-R model; sentiment is optional structured judge output,
which keeps the implementation small before dashboard validation.

## Policy-based quality benchmark

`callverse/quality/benchmark.py` contains 15 compact policy-based regression
fixtures. It is not a human-labeled dataset and is not statistical ground truth.
The cases cover:

- grounded and invented tracking status;
- safe nonexistent-order escalation;
- correct refund approval interruption and unauthorized refund execution;
- relevant-but-incomplete and irrelevant responses;
- followed and skipped address-change procedure;
- damaged-item escalation;
- empathetic and dismissive angry-customer handling;
- cancellation procedure;
- unsupported delivery promise;
- safe low-confidence fallback.

Expected properties focus on deterministic flags, review requirements, and
relative ordering of matched good/bad cases. Offline validation verifies that:

- grounded tracking scores above invented tracking;
- proper refund approval scores above unauthorized execution;
- a relevant answer scores above an irrelevant answer;
- followed address-change procedure scores above a skipped procedure;
- empathetic angry-customer handling scores above dismissive handling;
- safe escalation receives no hallucination penalty;
- weighted arithmetic matches the configured weights;
- hard flags survive an artificially favorable fake judge.

The benchmark uses an injected deterministic structured judge for nuanced score
fixtures. Those scores are test fixtures, not real LLM results or human labels.

## Dashboard-ready aggregation

`aggregate_quality` calculates, without a database:

- total, evaluated, and unavailable interaction counts;
- average overall quality;
- average score for each of the six dimensions;
- supervisor-review count;
- quality/compliance flag counts by severity;
- unresolved and escalated counts;
- sentiment distribution where sentiment is available.

These outputs prepare Phase 8 without adding UI or persistence in this phase.

## Open-source adaptation and attribution

The reference inspected was
[ANI-IN/Call-Center-Intelligence-System](https://github.com/ANI-IN/Call-Center-Intelligence-System)
at commit
[`fed4b610742c1337147281fcec8f2cdcc0a79be5`](https://github.com/ANI-IN/Call-Center-Intelligence-System/tree/fed4b610742c1337147281fcec8f2cdcc0a79be5).
The inspected repository contains an MIT license with copyright © 2026 Animesh
Kumar.

CallVerse adapts the reference's patterns of typed bounded scores, structured LLM
output, explicit compliance flags, deterministic overall-score recomputation,
and manager-level quality summaries. The CallVerse delivery rubric, evidence
contract, guardrails, benchmark, and code are project-specific. MIT attribution
and license text are recorded in `THIRD_PARTY_NOTICES.md`.

Deliberately not reused: Whisper/audio processing, transcription, Gradio, the
upstream LangGraph, database, report generator, security pipeline, and full
application structure. CallVerse remains text-first.

## Bad-review risk limitation

Conversation-based bad-review prediction is not implemented. Olist has delivery
and review outcomes, Bitext has support language, and HelpPilot provides
synthetic/demo interactions, but these are not records for the same customers.
Row-wise joining them would create a scientifically invalid target relationship.
No XGBoost bad-review model is trained or implied. This should remain an
extension point until defensibly joined conversation/review data exists or a
future simulation assumption is explicitly labeled as synthetic.

## Differences from the presentation and limitations

The presentation's future-facing model proposals are implementation options,
not evidence that those models exist. Quality Analyst V1 uses the existing Groq
provider for optional structured judgment, deterministic Python guardrails, and
no additional local sentiment or language model. It does not include audio,
Whisper, XLM-R, an LLM judge benchmark at production scale, or a bad-review
predictor.

The rubric and weights are V1 expert definitions. The policy fixtures validate
software behavior, not real-world evaluator validity. Proper future validation
should review approximately 100–300 CallVerse conversations with one, preferably
two, human reviewers and measure:

- human/model overall-score correlation;
- per-dimension agreement;
- compliance-flag precision and recall;
- inter-rater agreement.

That human validation has not been performed and must not be claimed.
