# CallVerse Dashboard / Scenario Studio V1

**Project implementation status: CallVerse V1 complete research prototype.** The
recommended defense order is Simulate → Observe → Forecast → Plan workforce → Test a
same-seed decision → Inspect an interaction → Evaluate quality.

## Purpose

The Manager Control Room is the first visual management surface for CallVerse:

> Run a virtual support-center scenario, observe the operational impact, then
> test a decision before applying it.

It extends the inherited Streamlit application rather than creating another
frontend. CallVerse still has one application entry point (`app.py`), one SimPy
engine, one Customer Advisor, and one Quality Analyst.

## Navigation

The sidebar provides three roles:

1. **Customer Support** — the inherited HelpPilot chat experience;
2. **Staff / Approvals** — the inherited durable refund-approval queue;
3. **Manager Control Room** — operational simulation and selected-interaction
   inspection.

The Manager Control Room contains seven focused tabs:

- Scenario Studio
- Twin Monitor
- Compare Decisions
- Forecast
- Workforce
- Interaction Lab
- Quality

## Scenario Studio

The manager can select all seven existing presets: `normal_day`, `rainy_peak`,
`flash_sale`, `staff_shortage`, `customer_crisis`, `knowledge_failure`, and
`perfect_storm`.

Active controls that currently affect a run are:

- simulation seed;
- available agents;
- demand multiplier;
- simulation duration;
- prototype or calibrated policy.

The UI uses the existing `ScenarioConfig`, `run_simulation`, `DEFAULT_POLICY`,
and calibrated support profile. It does not create a second simulator or
scenario schema. Runs happen only after **RUN DIGITAL TWIN** is pressed and are
kept in Streamlit session state.

The following scenario fields are visible under **Scenario context — not yet
causally connected** because the current queue engine does not use them:

- external condition;
- late-delivery rate;
- knowledge-base state;
- AI-advisor enabled state.

They are never presented as active queue levers.

## Operational KPIs and Twin Monitor

The result cards display only values actually produced by `SimulationResult`:

- generated contacts;
- completed contacts;
- average waiting time;
- SLA attainment;
- abandonment rate;
- occupancy;
- average handling time.

Unavailable values display `N/A`; the dashboard does not replace missing FCR,
customer satisfaction, cost, or forecast values with zero.

The Twin Monitor uses the engine's 15-minute snapshots for:

- queue size over simulated time;
- busy agents over simulated time;
- cumulative completed and abandoned contacts.

Actual intent and persona counts are shown as bar charts. Native Streamlit
charts are used; no additional visualization framework was added.

## Scenario warnings

Warnings are deterministic descriptions, not learned recommendations:

- occupancy above 85% — operating near saturation;
- abandonment above 15% — severe abandonment;
- SLA below 80% — poor SLA attainment;
- ending queue above the starting queue — backlog remains at the horizon.

These thresholds are explicit in `callverse/dashboard/view_models.py`.

## Manual before/after comparison

After a completed run, a manager can change available agents and press
**RUN BEFORE VS AFTER**. The AFTER configuration preserves:

- scenario and all non-staffing fields;
- random seed;
- demand multiplier;
- duration;
- prototype/calibrated policy.

The comparison shows generated contacts, average wait, SLA, abandonment,
occupancy, and AHT with deltas. Percent metrics use percentage-point deltas.
The UI labels these as simulated effects under the selected scenario and seed,
not guaranteed production causal effects. No optimizer or automatic staffing
recommendation is included.

## Interaction Lab

The Interaction Lab is visually and architecturally separate from the
high-volume Digital Twin. It runs only a selected customer request through the
existing `CallVerseCustomerAdvisor` and displays:

- classifier intent and confidence;
- accepted/fallback path;
- extracted order ID;
- Advisor response;
- tools and RAG citation IDs;
- resolution and escalation state.

With no Groq key, the UI visibly says **Offline deterministic demo mode** and
uses the existing deterministic demo runner. It never labels those responses as
live LLM output. When a key is available, a live mode becomes available but
runs only after an explicit button press.

The SimPy engine does not invoke the classifier, HelpPilot, or Quality Analyst
for every generated contact. Therefore operational scenario results never show
an invented “average scenario quality.”

## Forecast

The Forecast tab loads the selected Phase 9 artifact through the independent
forecasting service. It shows the final 24 hours of actual historical demand
beside 48 future half-hour predictions, predicted total contacts, the peak slot,
peak demand, and deterministic high-demand summaries. Labels explicitly separate
actual history from forecast values and do not imply prediction intervals.

Forecasts describe historical support-contact demand. They neither overwrite a
manager's scenario assumptions nor recommend staffing. Workforce decisions remain
separate from the forecast service.

## Workforce

The Workforce tab consumes the 48-point forecast through the independent
Workforce Manager service. It displays a clearly labelled Erlang-C analytical
staffing recommendation, the forecast curve, raw interval requirements, a
deterministically smoothed operational schedule, peak agents, total agent-hours,
target attainment, and capacity-shortfall warnings.

Managers configure the target service level, maximum occupancy, explicit forecast
buffer, maximum search capacity, and reduction hold. Fixed staffing and Erlang-C
plans are compared using theoretical metrics. A separate five-seed validation
table labels calibrated Digital Twin results as simulated values rather than
mixing them with theory. Dynamic capacity inside the frozen Twin is deliberately
deferred; the existing fixed-staffing Compare Decisions workflow remains available.

### Experimental PPO section

When the reviewed Phase 11 model, metadata, and held-out evaluation artifacts exist,
the Workforce tab also displays a clearly separated experimental PPO section. It
compares fixed, Erlang-C, and PPO strategies on the same held-out dates and stochastic
seeds, including mean and standard deviation for reward, abandonment, service level,
wait, occupancy, agent-hours, and staffing changes. A representative staffing
trajectory and exact training provenance are shown.

The UI deliberately reports the negative result: PPO converged to roughly 17 agents
in every interval and consumed 408 agent-hours/day. Its higher internal reward is not
presented as a business win because Erlang-C used roughly 118 agent-hours/day. The
section also warns that the compact batched RL environment is distinct from, and less
authoritative than, the calibrated continuous-time Digital Twin.

## Quality panel

Every selected interaction receives deterministic Quality Analyst guardrails.
Without a structured live judge, status is shown as unavailable/offline and no
six-dimension or overall score is displayed. Genuine hard flags and guardrail
observations remain visible.

If a live structured judge is explicitly run, the panel may show overall
quality, the six dimensions, flags, supervisor-review status, and sentiment.
Session aggregation reuses Phase 7 utilities and reports evaluated/unavailable
counts, averages only when scores exist, review counts, flag severities, and
sentiment distribution.

The 15 policy fixtures appear only as a clearly labeled **Quality regression
benchmark**. They are not presented as real customer quality.

## Reproducibility and export

Each run exposes scenario, seed, policy mode, agent count, demand multiplier,
and duration in a details panel. The compact JSON export contains:

- run configuration;
- request counts;
- final KPIs;
- deterministic warnings;
- before/after comparison when one exists.

It excludes secrets, hidden reasoning, chat text, and large request logs.

## Open-source UI references

Phase 8 inspected two references:

- ANI-IN/Call-Center-Intelligence-System at
  `fed4b610742c1337147281fcec8f2cdcc0a79be5`. Phase 7 already adapted and
  attributed its typed quality/observability patterns. Phase 8 reuses the
  CallVerse aggregation created from that work.
- thelostbong/Queueing-Simulation-and-Optimization-System at
  `762d29ee4aac62d5e184ad8a2b5f524bed60dcd5`, verified MIT, Copyright © 2026
  Nayeemuddin Mohammed. Its wait/utilization/SLA comparison presentation was
  inspected as conceptual inspiration only. No code, simulator, Erlang-C logic,
  plots, or generated figures were copied, so no new license notice was needed.

The authoritative operational results remain CallVerse's calibrated and
held-out validated simulator.

## Limitations and future extensions

The PPO policy is experimental and must not automate staffing. V1 still has no
validated cost model, weather/delivery causal mechanism, audio model, or bad-review
prediction model. The Manager Control Room remains decision support; the calibrated
Digital Twin and transparent analytical baselines remain authoritative.

## Final provider and failure behavior

The Customer view disables live chat when `GROQ_API_KEY` is absent while leaving the
Manager and offline Interaction Lab usable. If a requested live call fails, the UI
reports the provider error type and explicitly states that no deterministic response
was substituted. The Interaction Lab never silently labels offline output as live.

The Forecast view identifies its source as historical generic Technion contact-center
demand and excludes production, courier, delivery-event, and weather-causal claims.
The Workforce view names Erlang-C as the analytical staffing baseline, not an optimum.
The PPO section states **Operational recommendation: NOT ADOPTED**.
