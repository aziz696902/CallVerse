# CallVerse Demand Forecasting V1

## Business purpose and scientific scope

Demand Forecasting V1 estimates support-contact volume for the next 24 hours in
48 consecutive 30-minute intervals. It answers **how much support demand may be
coming**; it does not decide how many agents to schedule.

This is historical support-demand forecasting over the Technion Anonymous Bank
Call-Center Data. It is not delivery-incident forecasting and makes no causal
claim about delivery delays, rainfall, traffic, flash sales, reviews, or courier
events.

## Dataset and aggregation

The build reuses `callverse.calibration.technion.load_technion`, including its
documented schema checks, duplicate handling, timestamp validation, outcome
filtering, and duration-quality rules. No second Technion parser exists.

Accepted arrival timestamps are floored into half-hour slots and reindexed over
the complete 1999 calendar:

- source-clock range: 1999-01-01 00:00 through 1999-12-31 23:30;
- 17,520 half-hour slots;
- 435,785 accepted contacts;
- 4,691 zero-contact slots;
- zero missing timestamps after continuous reindexing;
- two complete zero-contact calendar days (1999-01-11 and 1999-01-12), or 96
  zero-filled half-hours, corresponding to dates absent from the 363 observed
  source dates.

The source does not provide a defensible timezone. Timestamps therefore remain
naive source-local clock values; no timezone conversion or daylight-saving
adjustment is invented.

The compact `demand_30min.csv` is the only serialized aggregate series. Raw
monthly files remain ignored and are not duplicated.

## Exploratory evidence

Mean half-hour demand is 24.87 contacts and variance is 739.16, giving a
variance-to-mean ratio of 29.72 and strong overdispersion. Useful demand
autocorrelations are 0.906 at lag 1, 0.849 at lag 2, 0.524 at the same slot one
day earlier, and 0.787 at the same slot one week earlier. The stored evaluation
artifact also records half-hour, weekday, monthly, and first/last-28-day means.
The last 28 days average 28.35 contacts per slot versus 21.64 in the first 28
days. This is evidence of a level/seasonal difference within the year, not proof
of a causal or monotonic trend.

## Chronological split

Rows are never shuffled:

| Period | Start | End |
|---|---|---|
| Train | 1999-01-01 00:00 | 1999-09-12 23:30 |
| Validation | 1999-09-13 00:00 | 1999-11-05 23:30 |
| Untouched test | 1999-11-06 00:00 | 1999-12-31 23:30 |

Candidate selection uses validation rolling-origin MAE only. The test period is
opened after selection and does not change the chosen model.

## Forecasting strategy and features

The LightGBM candidates are one-step count regressors rolled forward recursively
for 48 steps. Each prediction is appended to working history before the next
step, so the evaluation is genuinely multi-step rather than a relabelled
one-step result. Recursive forecasts can accumulate errors toward 24 hours.

Features available at forecast time are:

- half-hour slot, weekday, weekend, month, and cyclic slot/weekday encodings;
- lags 1, 2, 48, 96, and 336;
- shifted rolling means and standard deviations over 6, 48, and 336 slots.

Rolling calculations use `target.shift(1)` and never include the current target.
Recursive inference constructs features only from observed or previously
predicted history. Raw predictions are clipped at zero; dashboard formatting may
round them, while stored evaluation uses unrounded values.

## Models and validation selection

The five candidates use the same 54 validation origins, a 48-slot horizon, and
a daily stride:

| Candidate | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Previous-slot naive | 19.476 | 25.902 | 114.091% |
| Daily seasonal naive | 12.859 | 21.091 | 85.234% |
| Weekly seasonal naive | 8.259 | 14.010 | 61.496% |
| LightGBM regression | 6.418 | 10.844 | 75.477% |
| LightGBM Poisson | **6.174** | **10.099** | 71.550% |

LightGBM Poisson wins on the predefined validation MAE and is therefore the
selected model. The Poisson objective is reasonable for non-negative counts and
also produces the best validation mean Poisson deviance (4.127), but selection
was based on MAE rather than objective preference.

## Untouched test results

All models use 56 daily rolling origins and 2,688 common test observations:

| Candidate | MAE | RMSE | sMAPE |
|---|---:|---:|---:|
| Previous-slot naive | 23.329 | 30.302 | 115.975% |
| Daily seasonal naive | 14.650 | 24.584 | 81.705% |
| Weekly seasonal naive | 7.094 | 11.881 | 50.818% |
| LightGBM regression | 5.962 | 9.734 | 71.354% |
| Selected LightGBM Poisson | 6.149 | 10.018 | 70.724% |

Regression happens to have lower test MAE, but it does not replace Poisson: doing
so would select on the final test. sMAPE is high around the many zero/low-volume
overnight slots and should be read alongside MAE, RMSE, and Poisson deviance.

For the selected model, test MAE rises from 1.432 in the first two hours to 4.909
over hours 2–12 and 7.969 over hours 12–24. Test RMSE rises from 1.689 to 8.645
and 11.747. This is the expected recursive horizon degradation.

## Forecast service and artifact

`forecast_next_24h(history, artifact_path)` loads the selected versioned artifact,
requires at least 336 chronological history slots, and returns exactly 48 typed
`ForecastPoint` values spaced 30 minutes apart. Missing history or artifacts fail
with explicit errors.

The only model artifact is
`models/demand_forecast/selected_model.joblib`. It is small enough for the private
repository and is tracked with model metadata, size, SHA-256, LightGBM version,
and parameters. Rebuild deterministically with:

```powershell
uv run python -m callverse.forecasting.train
```

The rebuild parses the locally available source files, evaluates all candidates,
writes the compact aggregate and metrics, and retains only the selected final
model.

## Manager Control Room integration

The existing Streamlit Manager Control Room has one additional **Forecast** tab.
It shows actual recent history and future forecast as distinct series, predicted
24-hour total, peak half-hour slot and count, high-demand periods, and computed
alerts. It labels the view as a future forecast and does not provide a staffing
recommendation or automatically alter a scenario.

## Delivery-signal limitation and future work

No validated, time-aligned dataset currently connects Technion support contacts
to Olist late deliveries, weather, traffic, flash sales, courier incidents, or
customer reviews. V1 therefore performs no fake join and uses none of these as
features. They may become external scenario stressors or validated exogenous
features only if suitable linked data becomes available.

Phase 10 may consume the forecast through the typed service boundary to build a
Workforce Manager and transparent Erlang-C staffing baseline. That later phase,
not this model, answers the staffing question.
