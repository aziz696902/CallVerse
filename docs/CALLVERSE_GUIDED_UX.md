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

## Recommended demo

`Staff Shortage` is the recommended teaching scenario. **LOAD RECOMMENDED DEMO** sets
the controls to seed `404`, three agents, and calibrated mode. It does not run the
simulation, inject KPI values, or bypass **RUN DIGITAL TWIN**.

## Scientific labeling

All outputs remain simulated evidence. Same-seed comparisons are not production A/B
tests, interpretation targets are expert-defined, and suggested actions are analysis
guidance rather than guaranteed operational outcomes. Scenario mechanics, model
artifacts, datasets, and numerical parameters remain unchanged.

## Deferred to UX Phase 2

Decision Guidance & Before/After Storytelling may improve the Compare Decisions
journey later. Animated replay, avatars, queue animation, new charts, and redesigns of
Forecast, Workforce, Interaction Lab, and Quality remain out of scope.
