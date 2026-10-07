# CallVerse V1 Architecture and Implementation Matrix

**Status:** V1 complete research decision-support prototype. Not production-ready.

## Implemented architecture

```text
DATA / CALIBRATION
  Technion operational records ─┐
  Olist delivery/review records ├─> compact calibrated profiles and evidence
  Bitext support text ──────────┘
                    |
                    v
SCENARIO -> calibrated SimPy DIGITAL TWIN -> OPERATIONAL KPIs
                                                   |
                                                   v
historical contact series -> LightGBM FORECAST -> ERLANG-C WORKFORCE PLAN
                                                   |
                                                   v
                                    same-seed WHAT-IF TESTING

CUSTOMER -> TF-IDF INTENT CLASSIFIER -> HELPILOT LANGGRAPH ADVISOR
                                      -> SQLite TOOLS / Chroma RAG
                                      -> durable human APPROVAL / escalation
                                      -> QUALITY ANALYST

PPO -> separate workforce-training environment -> research comparison only
       (does not control or dynamically execute inside the frozen Digital Twin)
```

The primary operational story is scenario, Twin, KPIs, forecast, Erlang-C plan,
and same-seed decision testing. Customer interaction and Quality are inspected at the
individual-interaction level. PPO remains a visible negative experimental result.

## Implemented versus deferred

| Component | Original proposal | Final implementation | Status | Evidence / metric | Limitation |
|---|---|---|---|---|---|
| Digital Twin | Virtual support center | Calibrated SimPy discrete-event queue with seven scenarios | Implemented | Held-out calibration validation and deterministic regressions | Operational abstraction, not production telemetry |
| Customer simulation | Personas, demand, requests | Typed persona/intent mixes and stochastic contacts | Implemented | Seven presets execute with coherent KPIs | Individual simulated contacts do not invoke the LLM |
| Intent classification | Delivery-support routing, possibly transformer | Selected TF-IDF logistic regression; BERT-tiny evaluated | Implemented | Test macro-F1 1.000 on 1,004 Bitext rows | Templated/synthetic-like source; damaged/general unsupported |
| RAG Advisor | Tool-using support agent | Inherited HelpPilot LangGraph, Chroma retrieval/reranking, grounding review | Implemented | Live grounded tracking and policy retrieval smoke tests | Small smoke suite; no large human evaluation |
| Structured tools | Orders, tracking, refunds | Seeded SQLite tools with customer/order preflight | Implemented | Known order grounded; unknown order safely escalated | Seed data, not enterprise integration |
| Human approval | Safe sensitive actions | Durable refund interruption and staff resume flow | Implemented | Live refund stopped at approval before execution | No real payment/refund backend |
| Quality | Six dimensions and safety | Deterministic guardrails plus optional structured Groq judge | Implemented | Three live parses; adversarial hard-flag tests pass | LLM judgment is not human ground truth |
| Bad-review risk | Predict low review outcome | No model; evidence is not joined at conversation/outcome grain | Deferred | Limitation explicitly documented | Olist/Bitext/HelpPilot records cannot be validly joined |
| Demand forecasting | Future support load | Rolling 48-step LightGBM Poisson historical-demand forecast | Implemented | Test MAE 6.149; RMSE 10.018 | No delivery/weather causal inputs or prediction intervals |
| Workforce | Staffing recommendation | Calibrated Erlang-C analytical plan with buffer/occupancy constraints | Implemented, preferred baseline | Example: 42.5 agent-hours, analytical target 48/48 | Simplified M/M/c, no abandonment or shift constraints |
| PPO | Learned staffing policy | SB3 PPO in a separate calibrated workforce environment | Experimental, not adopted | 408 agent-hours/day versus Erlang-C mean 118.1 | Severe overstaffing; environment differs from Twin |
| Audio | Speech input/transcription | None | Deferred | No artifact or claim | Outside evidence and compute scope |
| Weather | Disruption-driven demand | Scenario context only | Deferred as causal model | Dashboard labels it non-causal | No joined weather/contact dataset |

## Scientific boundaries

- Technion describes generic bank contact-center mechanics, not delivery customers.
- Olist shows delivery/review association; it does not prove support-contact causation.
- Forecasting models historical contacts, not live courier, delivery-event, or weather effects.
- Erlang-C is an analytical M/M/c baseline, not the Digital Twin or a guaranteed optimum.
- PPO was cross-checked only under fixed staffing and was not dynamically validated in
  the frozen Digital Twin.
- Nuanced Quality scores are shown only after a real structured judge succeeds;
  deterministic hard guardrails remain authoritative.
