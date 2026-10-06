# CallVerse Workforce Manager V1

## Purpose and forecast-to-staffing flow

Phase 10 converts each of the 48 half-hour Demand Forecast points into an
explainable staffing baseline:

```text
forecast contacts → explicit safety buffer → hourly arrival rate
→ Erlang-C staffing sweep → minimum feasible agents → reduction hold
→ 24-hour operationalized schedule
```

It answers how many pooled agents to plan under stated analytical assumptions.
It is not a live scheduler, employee roster, or learned workforce policy.

## Erlang-C assumptions

Erlang-C is an M/M/c approximation with Poisson arrivals, exponential service,
identical agents, a stationary interval arrival rate, one pooled queue, and no
abandonment. CallVerse violates several assumptions: its calibrated Twin uses an
empirical service distribution, intent scaling, empirical patience, abandonment,
and time-varying arrivals. Erlang-C is therefore a transparent staffing heuristic;
the existing SimPy Digital Twin remains authoritative.

## Adapted open-source reference

The implementation substantially adapts queueing and staffing structures from
`thelostbong/Queueing-Simulation-and-Optimization-System`, exact revision
`762d29ee4aac62d5e184ad8a2b5f524bed60dcd5`, MIT, Copyright © 2026 Nayeemuddin
Mohammed. Adapted concepts are offered load, Erlang-C waiting probability,
theoretical expected wait, staffing sweeps, and minimum SLA staffing.

CallVerse replaces the reference's direct factorial implementation with the
stable Erlang-B recurrence and adds typed contracts, service-level probability,
occupancy, buffers, smoothing, shortfall state, and Twin validation. Its simulator,
plots, figures, time-dependent application, and €28/hour assumption were not
copied. The full MIT notice is preserved in `THIRD_PARTY_NOTICES.md`.

## CallVerse service rate and SLA

The default mean AHT is **3.181917 minutes**, loaded from
`data/processed/calibration/support_center_profile.json` at
`service_time_minutes.mean`. This gives **18.856555 contacts/hour/agent**. It is
the validated overall empirical service mean; using one mean in M/M/c remains an
approximation because the Twin scales handling distributions by intent.

The wait threshold is **2.0 minutes**, the same `sla_target_minutes` used by the
Digital Twin. The default target service level of **80%** is an expert-defined,
configurable V1 planning assumption, not a learned value. The default maximum
occupancy is likewise an explicit **85%** operational assumption.

## Stable mathematics and staffing search

All rates use contacts/hour. Forecast contacts are divided by the half-hour
interval length, so 10 buffered contacts in 30 minutes becomes 20 contacts/hour.
Offered load is `arrival_rate / service_rate`. Erlang-B recurrence avoids powers
and factorial overflow; Erlang C is derived from the stable blocking probability.

Zero demand returns zero utilization/wait and 100% service level. An unstable
`rho >= 1` returns an explicit unstable result with unavailable expected wait.
The search tests every configured integer staffing level and selects the first
that satisfies both service level and occupancy. If none through `max_agents`
succeeds, it returns the maximum evaluated candidate with
`capacity_shortfall=true`; it never presents the cap as a solved recommendation.

## Forecast buffer and uncertainty

Phase 9 has no prediction intervals. The configurable 0–30% dashboard safety
buffer is a planning assumption applied visibly before rate conversion, not a
confidence interval.

## Schedule smoothing

The plan retains both raw minimum staffing and operationalized staffing. The
default two-interval reduction hold requires lower demand to persist before a
staffing reduction. Increases happen immediately. This deterministic hysteresis
only delays reductions; it never conceals the raw Erlang-C requirement or creates
a complex optimizer.

## Agent-hours and optional cost

The primary cost measure is agent-hours. Each half-hour contributes
`agents × 0.5`. Optional `cost_per_agent_hour` exists but defaults to `None`, so no
currency or labor price is presented as CallVerse truth.

## Fixed baseline

The default fixed baseline is the rounded average agent requirement of the
operationalized Erlang-C plan, used unchanged for all 48 intervals. A manager may
instead enter a fixed count. Comparison reports agent-hours, average/peak agents,
target-attainment intervals, mean modeled utilization/service level, and
shortfall intervals. This is a defensible baseline, not a deliberately weak one.

## Actual Phase 9 plan

Using the selected Phase 9 forecast, default 10% buffer, 80% service target, 85%
occupancy cap, and two-interval reduction hold produces:

- 251.744 forecast contacts;
- 1–5 recommended agents;
- 1.771 average agents;
- 42.5 agent-hours;
- six staffing changes;
- peak staffing at 19:00 and 19:30;
- zero analytical shortfall intervals;
- all 48 intervals meeting both analytical targets.

The rounded-average fixed baseline is two agents: 48 agent-hours, 36/48 target
intervals, 30.6% mean modeled utilization, and 77.6% mean modeled service level.
The variable plan uses 5.5 fewer agent-hours while meeting all analytical targets,
but this is an analytical comparison rather than a guaranteed production saving.

## Independent Digital Twin validation

`data/processed/workforce/erlang_c_validation.json` contains four fixed-capacity
cases—low, normal, high, and near saturation—with seeds 101, 202, 303, 404, and
505. Each simulation runs the existing calibrated Twin for 480 minutes. No
calibration, patience, or service distribution is modified to force agreement.

At low, normal, and high demand, the minimum Erlang-C recommendations are compared
with the same fixed agent counts in the Twin. The near-saturation case deliberately
uses the smallest stable integer above offered load. Differences are expected:
the Twin abandons contacts and its realized AHT is about 3.77–4.00 minutes under
the scenario's intent mix, while Erlang-C uses the overall 3.182-minute mean and
assumes no abandonment. Near saturation, Erlang-C predicts 9.35 minutes wait and
29.5% SLA, while the Twin averages 1.60 minutes and 70.6% SLA with 24.5%
abandonment; abandonment removes queued demand and materially changes observed
wait/SLA.

## Dynamic staffing decision

Dynamic 30-minute capacity changes inside the Twin are deliberately deferred.
The engine uses a fixed-capacity `simpy.Resource`; safe reductions would require
new capacity semantics that preserve customers in service, queued requests,
occupancy accounting, snapshots, and seeded regression behavior. This is too
invasive for the frozen calibrated Twin in Phase 10. Representative fixed
recommendations are independently validated instead, and the existing manual
same-seed fixed-staffing comparison remains available.

## Dashboard integration

The seventh Manager Control Room tab, **Workforce**, exposes planning assumptions,
summary cards, separate demand and staffing charts, raw and smoothed requirements,
fixed-vs-variable comparison, deterministic explanations, capacity warnings, and
a separately labelled five-seed Twin validation table. There is no fake dynamic
simulation button.

## Limitations and Phase 11

The forecast comes from old Technion banking contacts and has no prediction
intervals or delivery/weather causal inputs. Erlang-C has no abandonment, agent
skills, shift legality, breaks, hiring constraints, employee identities, or
skill-based routing. Its interval-stationary arrival model approximates a varying
day and one pooled agent type.

Phase 11 may compare a PPO experimental policy with the same forecast inputs,
constraints, seeds, Digital Twin outcomes, and fixed/Erlang-C baselines. PPO must
earn improvement on held-out simulations rather than replacing this transparent
baseline by assumption.
