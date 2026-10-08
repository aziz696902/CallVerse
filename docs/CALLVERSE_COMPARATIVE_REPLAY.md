# CallVerse Comparative Simulation Replay

## Purpose and fixed comparison contract

Comparative Replay places two completed Digital Twin trajectories on one synchronized
timeline. It supplements the existing single-run replay and Compare Decisions; it is
not a new simulator or live production telemetry.

- **FIXED STAFFING BASELINE** uses the calibrated Digital Twin with the initial fixed
  staffing level.
- **CALLVERSE-ASSISTED DECISION** uses the same Digital Twin and seeded workload, with
  the tested staffing decision applied.

Both sides use the same scenario, seed, duration, demand multiplier, simulator mode,
intent mix, persona mix, and remaining scenario settings. Only available advisors may
differ. The assisted staffing level applies from simulation start; there is no dynamic
mid-run intervention. "CallVerse-assisted" does not imply that every model is active,
that the tested decision is optimal, or that an operational outcome is guaranteed.

Using the same seed reduces random variation so the staffing decision can be compared
under matched simulated demand. This is a controlled simulation comparison, not a
production A/B test or causal estimate.

## Primary validated teaching demo

The primary setup is Staff Shortage, seed 404, calibrated mode, 480 simulated minutes,
demand multiplier 1.15, and 3 baseline to 5 assisted advisors. Five is a tested
demonstration choice, not a globally optimal staffing claim.

The builder executes each existing simulation path once. It requires the two snapshot
timestamp sequences to be exactly equal and raises an error rather than interpolating
or silently pairing different times. Its 33 immutable frames have a 15-minute cadence.
Changing a setup field makes the completed comparison stale until it is rebuilt.

## Playback and operational view

Play, Pause, Restart, manual scrub, and 0.5x/1x/2x/4x/8x speed controls consume only
stored frames. At 1x, each 15-minute simulated interval is displayed for two wall-clock
seconds. Playback never invokes the simulator. Both sides always render from the same
frame index and simulated clock.

Advisor and waiting-contact markers are capped at ten. An explicit additional-count
label represents hidden markers, while the formatted numeric total remains
authoritative. This keeps the same view readable for both demonstrations.

## Phase 3 deterministic storytelling

The replay now includes a tested-decision card, permanent side definitions, an event
list, a current-frame comparison, one queue trajectory, and a final manager summary.
All story elements are deterministic; no LLM generates evidence or conclusions.

Allowed event markers are simulation start/end and first observed queue emergence,
sampled queue clearance, all-advisors-busy state, and MODERATE/HIGH/SEVERE snapshot
pressure for each side. Every event points to an actual stored frame. The official run,
for example, first shows the baseline queue, all advisors busy, and HIGH pressure at
08:45. It does not show a fictional "manager adds advisors" event.

The current-frame table separates two evidence types:

- current queue and busy/free advisors are **current snapshot** state;
- completed and abandoned contacts are **cumulative from simulation start**.

The queue-size chart plots the two values in every `ComparativeReplay.frames` item on
the shared simulated timeline. It performs no smoothing or interpolation and mixes no
other units into the chart.

At the final frame, the manager summary becomes prominent and reuses the existing
Compare Decisions KPI formatting and outcome classifier. For the official run it
reports `STRONGLY IMPROVED`, the actual KPI changes, and the resource trade-off of two
additional advisors. It does not implement a second outcome rule or infer monetary
cost. The result remains evidence from this matched simulated scenario and does not
establish a production optimum.

## Secondary Large Center Stress Test

The secondary option is a **simulated scalability demonstration**, not a separately
calibrated or validated production scenario. It scales the unchanged Staff Shortage
configuration proportionally from the official 3-to-5 staffing pattern. Seed 404,
duration, mode, scenario mechanics, and all non-staffing configuration remain fixed.
No alternative seed was tried.

The candidate grid was specified before selection as 5x, 8x, and 10x demand/staffing.
The selection rule was: choose the smallest candidate that generates more than 2,000
contacts, builds comfortably below one second locally, remains interpretable, avoids
pathological saturation on both sides, and does not make the assisted side trivially
idle. Selection was not based solely on the largest KPI improvement.

| Scale | Demand | Advisors | Generated | Baseline completed | Assisted completed | Baseline SLA | Assisted SLA | Baseline abandon | Assisted abandon | Baseline wait | Assisted wait | Baseline occupancy | Assisted occupancy | Final backlog B/A | Build runtime |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5x | 5.75 | 15 -> 25 | 2,117 | 1,726 | 2,085 | 81.28% | 100.00% | 17.71% | 0.57% | 1.01 min | 0.02 min | 93.70% | 67.70% | 1 / 0 | 0.229 s |
| 8x | 9.20 | 24 -> 40 | 3,358 | 2,845 | 3,330 | 89.06% | 100.00% | 14.44% | 0.12% | 0.77 min | <0.01 min | 93.99% | 67.00% | 4 / 0 | 0.419 s |
| 10x | 11.50 | 30 -> 50 | 4,211 | 3,590 | 4,172 | 87.04% | 100.00% | 13.94% | 0.14% | 0.76 min | 0.01 min | 93.07% | 65.58% | 4 / 0 | 0.456 s |

The selected case is **5x: demand 5.75, 15 baseline advisors, and 25 assisted
advisors**. It is already materially larger, meets the predefined threshold, has a
non-trivial 67.70% assisted occupancy, and minimizes unnecessary local work.

Five-run local medians for the selected case were 0.124 s for the baseline simulation,
0.123 s for the assisted simulation, 0.293 s for the complete comparison build, and
0.00000032 s for stored-frame access. The measured Streamlit response was 0.641 s for
build-and-render and 0.308 s for a scrub rerender on 2026-10-08. These timings describe
this PC, not a general service-level guarantee.

Scaling preserves the simulator's mechanisms but does not constitute empirical
validation at that organization size. It does not validate CallVerse for a real center
with the displayed contact volume, and it does not alter the existing scenario presets.

## Final demonstration instructions

1. Open **Twin Monitor > Fixed-staffing Comparative Replay**.
2. For the secondary capacity demonstration, select **PREPARE CAPACITY WHAT-IF**,
   inspect the tested decision card, then choose **BUILD COMPARISON**.
3. Play at 4x or scrub to 12:00. Explain that queue/busy values are snapshots while
   completed/abandoned values are cumulative.
4. Expand deterministic timeline events and point out that markers are stored-frame
   evidence, not generated narration.
5. Move to 16:00 and present the final outcome and +2-advisor trade-off.
6. If reproducibility is challenged, select **PREPARE SAME-STAFF CONTROL** and show
   exact 3-to-3 frame equality. Optionally use the Large Center Stress Test only for
   scalability.

## Frozen boundaries

Comparative Replay does not modify SimPy mechanics, calibration, arrival/service/
patience behavior, KPI formulas, existing presets, Forecast, Erlang-C, PPO, Advisor,
RAG, Quality, datasets, or artifacts. It does not invent per-frame SLA, occupancy,
average wait, AHT, satisfaction, cost, or intermediate points. Comparative Replay
development ends with this implementation.

The fixed 3-to-5 and large-center comparisons are explicitly **capacity what-ifs**:
they answer what happens when fixed capacity is added. The separate Fair Workforce
Comparison answers the **workforce intelligence** question of whether a forecast-aware
schedule can allocate a same-or-lower agent-hour budget more effectively. It does not
replace or alter these frozen replays.
