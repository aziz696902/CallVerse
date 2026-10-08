# CallVerse Dynamic Workforce Twin

## 1. Research question

The fixed 3-to-5 advisor replay is a useful **capacity what-if**, but it mainly shows
that adding capacity can improve service. It is not sufficient evidence that CallVerse
allocates a staffing resource intelligently.

The Dynamic Workforce Twin asks a stronger question:

> Can CallVerse allocate staffing capacity better over time, using the same or lower
> total staffing budget, than a simple fixed-staffing baseline?

The primary experiment compares two advisors fixed throughout 24 hours with the
existing Forecast-to-Erlang-C schedule. The budget is measured in agent-hours, not peak
advisor count.

## 2. Fixed versus scheduled staffing

The original `run_simulation` path remains the fixed mode and retains the existing
SimPy `Resource` behavior. Scheduled staffing is opt-in through
`run_scheduled_simulation`; no existing preset silently becomes dynamic.

Scheduled capacity is controlled by a CallVerse-owned FIFO gate built with public
SimPy events. It does not modify `Resource._capacity` or another private SimPy field.
The schedule changes only at discrete 30-minute boundaries.

When capacity increases, queued requests may begin service immediately. When it
decreases, already-running services continue with their original handling time. New
services remain blocked until the active count falls below scheduled capacity. No
contact is cancelled, requeued, reset, or dropped because of a reduction.

At a reduction boundary, `busy_agents` can temporarily exceed `available_agents`.
Snapshot semantics are therefore:

- `free_agents = max(available_agents - busy_agents, 0)`;
- `overhang_busy_agents = max(busy_agents - available_agents, 0)`.

Overall scheduled occupancy is busy advisor-minutes divided by scheduled capacity
minutes. It may exceed 100% in a pathological overhang experiment; this is retained
rather than hiding active work. Fixed-mode occupancy remains unchanged.

## 3. Schedule contract and agent-hour fairness

`StaffingSchedule` is immutable and validates positive integer staffing, one declared
interval length, continuous non-overlapping slots, a minute-zero start, and complete
horizon coverage. For each slot:

`agent-hours = advisors * interval minutes / 60`

The schedule total is the exact sum. The primary fixed baseline uses 48 half-hour slots
at two advisors, equivalent to true fixed staffing and exactly 48.0 agent-hours.

The existing default workforce plan recomputes to:

- 48 half-hour slots;
- minimum 1, average 1.770833, maximum 5 advisors;
- 42.5 agent-hours;
- 6 staffing changes;
- 48/48 analytical target-attainment slots.

The CallVerse budget is therefore 5.5 agent-hours lower, or 11.46% below the fixed
baseline. No monetary cost is inferred.

## 4. Forecast to Erlang-C to schedule to Twin

The schedule is generated upstream of the Dynamic Twin:

`existing LightGBM forecast -> default Erlang-C Workforce Manager -> immutable
30-minute schedule -> scheduled Digital Twin`

It uses the existing 80% service-level target, 85% maximum occupancy, 10% forecast
buffer, calibrated mean AHT, and two-interval reduction hold. The schedule was not
drawn manually, PPO is not used, and no parameter was changed after seeing the Dynamic
Twin result.

## 5. Explicit time and demand alignment

The forecast origin is 1999-12-31 23:30. Its points cover 2000-01-01 00:00 through
23:30 in 48 half-hour steps. Simulation minute zero maps exactly to the first forecast
point; slot `i` maps to elapsed minutes `[i*30, (i+1)*30)` and to forecast point `i`.

The 48 predicted contact values provide the Digital Twin's 48 arrival-profile weights.
The scenario demand multiplier is computed before simulation so expected 24-hour
arrivals equal the forecast total of 251.744 contacts. This is a direct slot alignment,
not an arbitrary mapping to the normal 08:00 scenario clock.

Both runs use that same arrival profile, seed 404, scenario configuration, customer
mix, calibrated service distributions, and calibrated patience distributions. The
engine already separates arrival, attribute/patience, and service random streams.
Consequently, all 248 realized contacts have identical arrival times, intents,
personas, patience draws, and handling draws across the two policies. Only capacity
admission differs.

## 6. Main experimental result

| Metric | Fixed 2 advisors | CallVerse dynamic | Dynamic minus fixed |
|---|---:|---:|---:|
| Agent-hours | 48.0 | 42.5 | -5.5 |
| Generated | 248 | 248 | 0 |
| Completed | 174 | 228 | +54 |
| SLA | 60.92% | 90.79% | +29.87 pp |
| Abandonment | 29.84% | 8.06% | -21.77 pp |
| Average wait | 2.66 min | 0.54 min | -2.13 min |
| Occupancy | 24.81% | 36.97% | +12.15 pp |
| Final backlog | 0 | 0 | 0 |

For this predefined simulated experiment, the CallVerse schedule improves SLA,
abandonment, waiting, and completed contacts while using fewer scheduled agent-hours.
The deterministic classification is **IMPROVED WITH LOWER RESOURCE BUDGET**. This
supports a resource-allocation result for this simulation; it does not prove a
production optimum.

## 7. Real workforce-plan events

The six events are schedule instructions, not live manager interventions:

| Time | Planned change |
|---|---:|
| 17:30 | 1 -> 2 advisors |
| 18:00 | 2 -> 3 advisors |
| 18:30 | 3 -> 4 advisors |
| 19:00 | 4 -> 5 advisors |
| 20:00 | 5 -> 4 advisors |
| 23:30 | 4 -> 3 advisors |

The UI wording is “CallVerse staffing plan changes to N advisors.”

## 8. Controls

- Fixed Staff Shortage, seed 404, three advisors remains exactly 394 generated, 289
  completed, 97 abandoned, 64.73% SLA, 24.62% abandonment, 2.11-minute wait,
  85.05% occupancy, and final backlog 5.
- A constant scheduled `[3, 3, ...]` Staff Shortage run is exactly equal to fixed mode
  for final counts, KPIs, snapshots, request records, busy minutes, and capacity minutes.
- A focused 5-to-2 reduction proves five in-service contacts finish normally, exposes
  three overhang-busy advisors, and blocks new starts until capacity permits.
- A focused 2-to-5 increase releases three already-waiting contacts exactly at the
  schedule boundary.

No neutral 42.5-hour secondary baseline was added. Alternating one and two advisors or
placing a half-hour arbitrarily would itself introduce a temporal allocation policy;
without a defensible independent rule, it would not be a scientifically neutral
control. The simple fixed two-advisor baseline is predefined, interpretable, and uses
more—not less—total budget.

## 9. Performance

Local medians on 2026-10-08 were:

- fixed 24-hour Twin: 0.0134 seconds;
- constant scheduled 24-hour Twin: 0.0117 seconds;
- dynamic workforce Twin: 0.0122 seconds;
- complete fair comparison: 0.0326 seconds;
- snapshots: 97 per run at the existing 15-minute cadence.

The constant scheduled timing was within normal measurement noise of fixed mode, so no
measurable schedule-application overhead was observed. These are local measurements,
not service-level guarantees.

## 10. Scientific limitations

- Scheduling decisions occur every 30 simulated minutes.
- Individual named employees, shift hand-offs, skills, labor law, breaks, and overtime
  constraints are not modeled.
- The forecast originates from historical generic Technion call-center data.
- Erlang-C remains an analytical M/M/c planning baseline.
- Schedule agent-hours do not add a monetary assumption.
- Same-seed common random numbers control generated workload, but staffing changes
  legitimately change waits, service admission, abandonment, and completion timing.
- The result is simulation evidence about resource allocation, not proof of a real
  organization outcome or a production optimum.

## 11. Visualization boundary

This phase exposes a basic 48-row table containing forecast and realized demand,
baseline and scheduled advisors, queue, busy/available/free/overhang staffing,
completed, and abandoned values. It also exposes deterministic staffing-change events.
Prompt 2 may improve how these already-proven data are visualized; this phase does not
add animated schedule bars, workforce-floor animation, Sankey diagrams, gauges, 3D,
or cinematic transitions.
