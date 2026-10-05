# CallVerse Digital Twin V1

## Scope

Digital Twin V1 is an offline SimPy discrete-event model of operational flow in a
customer-support center. One simulation minute represents one minute of center time.
The engine consumes the existing `ScenarioConfig` and produces validated aggregate
KPIs, request counts, bounded per-request records, and 15-minute time-series snapshots.

The simulation remains in memory. It does not call HelpPilot, Groq, LangGraph, Chroma,
or any other external service.

## Operational Flow

Requests arrive according to a Poisson-style process: inter-arrival times are sampled
from an exponential distribution. The explicit prototype baseline is 0.75 requests per
simulated minute, multiplied by the scenario's `demand_multiplier`. The scenario seed
makes the full run reproducible, and its duration sets the time horizon.

Each generated request uses the scenario's request-intent and customer-persona
probability mixes. It is represented by the existing `SupportRequest` contract. A
finite SimPy resource represents the scenario's available advisors. A request starts
immediately when an advisor is free or waits in the shared first-come, first-served
queue.

While waiting, a request races advisor availability against its sampled customer
patience. If patience expires first, it leaves the queue as abandoned and cannot later
receive service. If the advisor is acquired first, its waiting time is recorded and it
receives service for its pre-sampled handling duration.

## Prototype Behavior Assumptions

All stochastic values in `callverse/simulation/policies.py` are provisional,
illustrative parameters—not research findings or calibrated operational estimates.
They are centralized so later dataset work can replace them cleanly.

- Tracking contacts have shorter triangular handling-time distributions.
- Refund and payment contacts are medium-length.
- Damaged-item and complaint contacts are longer.
- Loyal customers receive a higher prototype patience distribution.
- Unhappy and at-risk customers receive lower prototype patience distributions.
- The SLA target is service starting within 2 simulated minutes.
- Compact snapshots are recorded every 15 simulated minutes.

Separate random streams generate arrival timing, customer/request attributes, and
handling durations. This keeps controlled comparisons reproducible and prevents a
staffing change from silently changing the synthetic request stream.

## Derived KPIs

The engine derives rather than fabricates:

- generated, completed, abandoned, and remaining request counts
- average observed waiting time for served or abandoned requests
- average handling time for requests completed inside the horizon
- abandonment rate
- advisor occupancy from busy advisor-minutes divided by available capacity-minutes
- SLA from the proportion of served requests starting within the configured threshold

Requests still queued or in service when the horizon ends are counted as remaining.
First-contact resolution, customer satisfaction, and operating cost stay `None`
because Digital Twin V1 has no defensible way to calculate them.

## Scenario-Driven Behavior

The current engine uses scenario duration, random seed, demand multiplier, available
agents, request-intent mix, customer-persona mix, and the AI-automation flag recorded
in results. The automation flag has no queue effect yet.

`external_condition`, `late_delivery_rate`, and `knowledge_base_state` remain preserved
scenario context. Their independent behavioral effects are deferred: the Phase 2
presets already express operational traffic differences through demand and intent
mixes, and inventing extra multipliers would double-count or fabricate effects.

All seven presets use the same engine. No scenario-specific simulation code exists.

## Outputs and Comparison

`SimulationResult` contains validated counts and KPIs, busy and available-capacity
advisor-minutes, intent/persona totals, bounded request records (maximum 2,000 by
default), and compact queue snapshots. The comparison utility runs multiple scenarios
with an optional common seed. Run the quick offline demo with:

```bash
python -m callverse.simulation.demo
```

## Not Yet Modeled

Digital Twin V1 models operational flow only. It does not yet model:

- actual conversation quality or semantic resolution
- real RAG success or failure
- customer satisfaction or bad-review risk
- forecasting
- dynamic workforce decisions or optimization
- reinforcement learning
- live advisor automation effects

The initial arrival, handling, and patience distributions must be calibrated and
validated later with appropriate contact-center datasets before real-world use.
