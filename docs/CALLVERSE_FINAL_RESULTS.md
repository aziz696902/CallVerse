# CallVerse V1 Final Verified Results

**Project status:** V1 complete research decision-support prototype. This summary uses
the committed machine-readable calibration, classification, forecasting, workforce,
RL, and final smoke artifacts. Evaluation domains remain separate.

## Calibration and Digital Twin

The Technion Anonymous Bank data is used only for generic contact-center mechanics.
Of 444,448 raw records, 435,785 were usable. The calibrated overall mean service time
is 3.182 minutes, mean recorded wait is 1.645 minutes, and queued-call abandonment is
24.55%.

The chronological held-out period contains 94,934 calls. Its mean service time is
3.333 minutes and queued-call abandonment is 23.28%, compared with 3.138 minutes and
24.88% in calibration. Empirical quantiles were retained because the observed service
distribution is heavy-tailed. Olist lateness/review results are associative only and
are not joined to Technion contacts.

All seven calibrated scenario presets executed successfully with their declared seeds.
The official examples are:

| Scenario | Seed | Contacts | SLA | Abandonment | Occupancy |
|---|---:|---:|---:|---:|---:|
| Normal day | 101 | 352 | 100.00% | 0.00% | 34.60% |
| Staff shortage | 404 | 394 | 64.73% | 24.62% | 85.05% |
| Perfect storm | 707 | 934 | 27.92% | 53.96% | 98.42% |

These are deterministic simulation outcomes, not production measurements.

## Intent classification

The evaluation used 1,004 held-out Bitext rows across six supported labels.

| Model | Test macro-F1 | Test accuracy | Decision |
|---|---:|---:|---|
| Majority class | 0.0922 | 0.3825 | Baseline only |
| TF-IDF + logistic regression | 1.0000 | 1.0000 | Selected |
| BERT-tiny | 0.9841 | 0.9920 | Not selected |

TF-IDF was selected by the predeclared validation rule. Perfect scores reflect the
highly templated/synthetic-like dataset and must not be generalized to production
customer language. Damaged-item and general language remain explicit fallback cases.

## Historical demand forecasting

The forecasting series contains 17,520 half-hour slots and 435,785 contacts. Models
were selected using chronological rolling-origin 48-step validation.

| Model | Test MAE | Test RMSE | Test sMAPE | Poisson deviance |
|---|---:|---:|---:|---:|
| Previous slot | 23.329 | 30.302 | 115.98% | 72.291 |
| Daily seasonal naive | 14.650 | 24.584 | 81.70% | 151.232 |
| Weekly seasonal naive | 7.094 | 11.881 | 50.82% | 8.883 |
| LightGBM regression | **5.962** | **9.734** | 71.35% | **2.963** |
| LightGBM Poisson | 6.149 | 10.018 | **70.72%** | 3.088 |

LightGBM Poisson was selected on validation, not retrospectively on test MAE. The
final 48-point example predicts 251.744 contacts, peaks at 24.769 contacts at 19:00,
and covers 2000-01-01 at 30-minute resolution. It is historical support-demand
forecasting, not a courier, delivery-event, or weather-causal forecast.

## Workforce Manager

The Phase 9 forecast was translated using calibrated mean AHT 3.182 minutes,
two-minute SLA threshold, 80% analytical service target, 85% occupancy cap, 10%
forecast buffer, and a two-interval reduction hold.

| Strategy on final forecast | Agent-hours | Average agents | Target slots |
|---|---:|---:|---:|
| Fixed 2 agents | 48.0 | 2.000 | 36/48 |
| Erlang-C operationalized plan | 42.5 | 1.771 | 48/48 |

The Erlang-C result is an analytical M/M/c recommendation, not the Digital Twin or a
guaranteed optimum. Fixed-staffing comparisons against the calibrated Twin showed
similar capacity direction but expected differences from abandonment, empirical
handling times, and time-varying arrivals. Dynamic staffing remains unsupported in the
frozen Twin.

The official staff-shortage what-if changed only agents from 3 to 5 under seed 404.
Simulated SLA improved from 64.73% to 98.41%, abandonment fell from 24.62% to 4.31%,
and mean wait fell from 2.11 to 0.20 minutes. This is a same-seed simulated effect, not
a production causal estimate.

## PPO workforce research

PPO was evaluated in its separate 48-step workforce environment over 36 held-out
episodes per strategy with common dates and stochastic seeds. These values must not be
mixed with the continuous-time Twin metrics above.

| RL-environment strategy | SLA | Wait | Abandonment | Agent-hours | Occupancy |
|---|---:|---:|---:|---:|---:|
| Fixed | 12.84% ± 6.45% | 12.21 ± 2.85 min | 19.64% ± 11.16% | 120.0 | 48.96% |
| Erlang-C | 14.29% ± 3.77% | 10.47 ± 1.61 min | 2.91% ± 2.65% | 118.1 ± 43.9 | 45.35% |
| PPO | 30.82% ± 14.40% | 4.94 ± 1.60 min | 0.21% ± 0.57% | 408.0 | 19.64% |

PPO converged to roughly 17 constant agents. Its lower wait/abandonment was purchased
with severe overstaffing. **PPO is not adopted; Erlang-C remains the operational
analytical baseline.** PPO was not dynamically validated inside the Digital Twin.

## Advisor and Quality integration

A bounded Groq integration smoke suite covered six conceptual cases. Five live Advisor
executions succeeded; the unknown-order case was safely stopped by deterministic
preflight without invoking the LLM. Verified behaviors included:

- known tracking used `get_order` and `get_tracking` and preserved seeded status;
- unknown `ORD-9999` did not receive an invented state and was escalated;
- refund handling retrieved policy, created only a draft, and paused for approval;
- policy retrieval ran, but the response escalated rather than directly answering;
- damaged-item language used the documented fallback and retrieved order/policy evidence;
- an angry complaint used verified processing state and avoided promising a ship date.

One damaged-item attempt hit a `RateLimitError`; a later bounded retry succeeded. This
is an integration smoke test, not a benchmark.

Three live structured Quality evaluations parsed all six dimensions within 1–5. The
policy, angry-complaint, and damaged-item cases scored 2.75, 3.20, and 4.00. Supervisor
review was required for the first two. Deterministic adversarial regressions separately
verified that invented order status is flagged, unauthorized refunds are capped, and a
favorable judge cannot erase hard flags. Live LLM scoring is not human ground truth.

## Final limitations

- Bank contact-center mechanics and delivery-support semantics come from separate domains.
- No joined late-delivery → support-contact dataset exists.
- Delivery and review evidence is associative, not causal.
- No weather causal model or live production forecast exists.
- Bitext is templated/synthetic-like and does not represent broad real traffic.
- The live Advisor suite is small and has no large-sample human evaluation.
- Quality LLM judgments are not human ground truth.
- No valid joined data supports a bad-review prediction model.
- Erlang-C uses simplified queue assumptions and omits shift/labor constraints.
- The frozen Twin does not support dynamic interval staffing.
- The PPO environment differs materially from the Twin.
- Audio and speech processing are deliberately deferred.
