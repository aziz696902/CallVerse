# CallVerse Guided UX

## Purpose and manager journey

UX Phase 1 adds a presentation layer to the frozen V1 Manager Control Room. It guides
a manager through **Choose scenario → Observe → Forecast → Plan → Test → Inspect
interaction → Review quality** while retaining the existing seven tabs and every
technical KPI.

The Scenario Studio now explains the business situation before a run. After a real
Digital Twin run, a deterministic summary translates its KPIs into an operational
status, target checks, reasons, and one next analysis step.

## Scenario cards

Each of the seven presets has separate typed presentation metadata:

- title and short description;
- situation and manager question;
- expected behavior and suggested action;
- descriptive risk level (`Low`, `Moderate`, `High`, or `Severe`);
- operational objectives and a scientific scope note.

Risk is descriptive UI guidance based on the preset and its teaching purpose. It is
not a learned score and does not modify `ScenarioConfig`.

Rain, delivery disruption, late-delivery rate, and knowledge-base state are context
fields in the current queue simulation. The cards do not claim that weather or these
context fields causally change demand or queue outcomes. Demand, staffing, request
mix, and persona mix are identified separately because the engine uses them.

## Result status rules

The expert-defined V1 managerial targets are:

- SLA at least 80%;
- abandonment below 10%;
- occupancy below 85%.

These are interpretation targets, not learned thresholds. Target equality passes for
SLA, but abandonment and occupancy use strict “below” objectives. Occupancy at or
above 85% is displayed as a caution.

The pure interpreter assigns:

- `HEALTHY`: all three targets pass;
- `UNDER PRESSURE`: one target misses or reaches caution;
- `OVERLOADED`: at least two targets miss or reach caution;
- `CRITICAL`: SLA is below 40%, abandonment is at least 40%, or occupancy is at
  least 97%.

Reasons may also mention average wait above the run's SLA wait threshold or a final
queue backlog. These observations do not change the status rules. Healthy runs point
to Forecast, one-target pressure points to Twin Monitor, and overloaded/critical runs
point to a same-seed staffing comparison.

## Jury starting scenario

`Staff Shortage` is the jury starting scenario. **LOAD JURY STARTING SCENARIO** sets
the controls to seed `404`, three agents, and calibrated mode. It does not run the
simulation, inject KPI values, or bypass **RUN DIGITAL TWIN**.

## Scientific labeling

All outputs remain simulated evidence. Same-seed comparisons are not production A/B
tests, interpretation targets are expert-defined, and suggested actions are analysis
guidance rather than guaranteed operational outcomes. Scenario mechanics, model
artifacts, datasets, and numerical parameters remain unchanged.

## UX Phase 2 — Decision Guidance & Before/After Storytelling

Compare Decisions now presents every completed staffing test as **Problem → Proposed
Action → Simulated Effect → Manager Conclusion**. It inherits the latest Scenario
Studio run, explicitly identifies the agent-count change, and lists the scenario,
seed, simulator mode, duration, demand multiplier, and remaining configuration as held
constant.

Both runs use the same seed and scenario configuration to reduce random variation and
make the staffing change easier to compare. This remains a simulated controlled
comparison, not a production A/B test or a claim of real-world causality.

### Delta and direction rules

- SLA and occupancy use percentage-point deltas, not relative percentages.
- abandonment also uses percentage-point deltas, with lower values preferred;
- average wait uses absolute minutes, with lower values preferred;
- completed and remaining contacts use absolute count differences;
- completed contacts are useful context but do not alone prove a better decision;
- occupancy is assessed relative to the 85% cap, so reducing already-healthy occupancy
  is neutral rather than automatically better.

Before/after checks reuse the Phase 1 targets: SLA at least 80%, abandonment below
10%, and occupancy below 85%.

### Deterministic outcome categories

The primary material-change checks use SLA, abandonment, and wait. A change of at
least two percentage points for SLA/abandonment or 0.25 minutes for wait is material.

- `STRONGLY IMPROVED`: a previously unhealthy center becomes healthy, at least two
  primary metrics materially improve, and none materially worsen;
- `IMPROVED`: center status improves to healthy/under pressure with a material service
  gain, or primary service gains occur without material worsening;
- `MIXED TRADE-OFF`: service improves but the center remains overloaded/critical, or
  another primary metric materially worsens;
- `LITTLE CHANGE`: no primary metric changes materially;
- `WORSENED`: primary service metrics materially deteriorate without an offsetting
  improvement.

The conclusion also states the added or removed advisor count without inventing a
monetary cost. One deterministic next step follows the outcome.

For the official Staff Shortage context, **PREPARE CAPACITY WHAT-IF DECISION** fills
the after value with five agents. It neither runs the comparison nor injects KPI
values.

Compare Decisions is manual what-if testing. Workforce remains the separate Erlang-C
analytical recommendation and neither feature labels the tested configuration as
optimal.

## UX Phase 3 — Navigation Hierarchy & Digital Twin Replay

### Navigation hierarchy

The application now lands on **Manager Control Room**, the primary CallVerse
experience. Its short introduction connects simulation, operational observation,
forecasting, staffing tests, and quality inspection.

Two supporting operational views remain available in the sidebar:

- **Customer Interaction Demo** demonstrates the deeper Advisor pipeline, including
  classification, tools, RAG, LLM response, escalation, and approval behavior.
- **Human Approval Queue** demonstrates human review of sensitive actions.

Interaction Lab and Quality remain inside Manager Control Room for compact manager
inspection. The separate Customer view is a deeper end-user conversation demo; it is
not another Advisor implementation.

### Replay architecture

Digital Twin Replay appears inside Twin Monitor after a scenario run. It reads the
existing immutable `TimeSeriesSnapshot` sequence from the latest `ManagerRun`; moving
the time-step slider selects a snapshot and never invokes the simulator again.

Each frame shows:

- human-readable simulated clock time and elapsed minute;
- waiting queue size;
- busy, free, and available advisors;
- cumulative completed and abandoned contacts;
- a deterministic descriptive queue-pressure level.

The pressure label uses only actual queue size and instantaneous busy/available
staffing: queue at least twice staffing is `SEVERE`, queue at least staffing is
`HIGH`, any queue or at least 85% busy is `MODERATE`, and otherwise it is `LOW`. This
is presentation guidance, not a learned score or cumulative occupancy metric.

Queue icons are capped at ten with a numeric remainder, so large queues cannot expand
the layout. Replay widget keys include scenario, seed, staffing, demand, duration, and
policy mode, preventing a slider position from leaking into a different run.

Snapshots do not store per-frame generated contacts, occupancy, SLA, or wait. The UI
does not synthesize them; final KPIs remain in Scenario Studio. Completed and
abandoned snapshot counters are explicitly labelled cumulative.

The replay is always labelled simulation replay rather than live telemetry. It
supplements the existing KPI cards and Twin Monitor charts. Frame construction is an
in-memory view operation and does not add meaningful simulation latency.

## UX Phase 4 — Final Demo Polish & Usability Freeze

The frozen manager journey is **Scenario → Twin → Forecast → Workforce → Compare →
Interaction → Quality**. The Manager Control Room is the primary defense surface;
Customer Interaction Demo and Human Approval Queue remain supporting views. The
recommended Staff Shortage path is visible in the page but never auto-runs an action.

Each stage now identifies the kind of evidence it presents:

- Scenario Studio and Twin Monitor show **simulated** Digital Twin evidence.
- Forecast shows a **historical ML forecast** of contact demand for 48 half-hour slots;
  it is not live weather data or a customer-satisfaction forecast.
- Workforce shows an **analytical Erlang-C baseline** derived from forecast demand;
  it is not an optimality claim or operational guarantee.
- Compare Decisions shows a **simulated same-seed comparison** and remains distinct
  from the analytical Workforce recommendation.
- Interaction Lab separates structured customer/order tools from RAG policy and
  procedure retrieval. The classifier routes the request, and the LLM writes a live
  response only when the provider is explicitly available and selected.
- Quality separates deterministic hard safety/compliance guardrails from optional
  LLM judgment of nuanced dimensions. LLM scores are not human ground truth.
- The PPO policy is labelled **EXPERIMENTAL — NOT ADOPTED** because its result relied
  on severe overstaffing in a simplified training environment that is not the
  calibrated Digital Twin.

Replay labels distinguish current state from cumulative flow. **Current snapshot
pressure** describes only the selected moment, whereas completed and abandoned
counters accumulate from the beginning of the run. A low-pressure final snapshot can
therefore coexist with substantial abandonment caused by earlier stress.

Final polish added concise KPI definitions, actionable empty states, explicit live
provider failure messaging, and a single official demo route. It did not change the
simulation, forecasting, staffing, Advisor, RAG, Quality, or RL mechanics.

| Phase | Focus | Status |
|---|---|---|
| UX Phase 1 | Guided Scenario Studio | Implemented |
| UX Phase 2 | Decision Storytelling | Implemented |
| UX Phase 3 | Replay + Navigation | Implemented |
| UX Phase 4 | Final Demo Polish | Implemented |
