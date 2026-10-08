# CallVerse Comparative Simulation Replay

## Purpose

Comparative Replay places two completed Digital Twin trajectories on one synchronized
manual timeline. It supplements the existing single-run replay and Compare Decisions;
it is not a new simulator and it is not live production telemetry.

## Two comparison sides

- **FIXED STAFFING BASELINE** is the reference run with a fixed advisor count.
- **CALLVERSE-ASSISTED DECISION** is the second run with a fixed staffing choice that
  a manager prepares or tests using CallVerse.

Both sides use the same existing Digital Twin, scenario, seed, duration, demand
multiplier, simulator mode, intent mix, persona mix, and remaining scenario settings.
Only available advisors may differ. “CallVerse-assisted” does not imply that every
model is active at each frame, that the decision is optimal, or that an operational
outcome is guaranteed.

The official teaching setup is Staff Shortage, seed 404, calibrated mode, three
baseline advisors, and five assisted advisors. Five is a tested demonstration choice,
not a globally optimal staffing claim.

## Synchronized timeline

The builder executes the baseline once and uses the existing staffing what-if path to
execute the assisted run once. It requires the two snapshot timestamp sequences to be
exactly equal; it raises an error rather than interpolating or silently pairing
different times. The resulting immutable frames contain both sides at one simulated
minute and clock time. Moving the manual selector reads a stored frame and does not
invoke either simulation again.

Changing any visible scenario, seed, staffing, demand, duration, mode, or decision
source makes stored frames stale and hides them until the comparison is rebuilt.

## Available per-frame evidence

`TimeSeriesSnapshot` stores exactly:

- simulated time;
- current queue size;
- busy advisors;
- completed contacts so far;
- abandoned contacts so far.

Available advisors comes from the completed run. Free advisors is the exact derived
value `available advisors - busy advisors`, consistent with the SimPy resource
invariant. Current snapshot pressure reuses the transparent existing queue/staffing
rule. Each frame also exposes its index, total frame count, simulated clock, elapsed
minutes, and the uniform cadence when one exists.

The snapshot does **not** store generated contacts so far, per-frame occupancy, SLA,
average wait, AHT, customer satisfaction, or cost. Comparative Replay does not invent
or interpolate them. Final SLA, abandonment, average wait, occupancy, completed
contacts, and backlog are read from the two normal completed results and formatted by
the existing decision-guidance logic.

Snapshot pressure describes the selected moment only. Completed and abandoned values
are cumulative and may reflect earlier stress even when current pressure is low.

## Scientific limitations

This is a same-seed simulated controlled comparison, not a production A/B test or a
causal estimate. Both runs keep staffing fixed throughout; there is no dynamic
mid-run capacity change. Shared randomness reduces variation but does not establish
real-world impact. The right-hand choice remains a tested decision, not an optimality
or forecasting guarantee.

## Phase 2 — Playback and operational view

Phase 2 adds a playback controller that consumes only the immutable synchronized
frame tuple. The controller stores frame index, playing/paused state, replay speed,
and end-of-replay state independently from the simulation data. Play advances exactly
one frame at a time, Pause preserves the selected frame, Restart returns to frame zero
without rebuilding either run, and manual scrubbing pauses playback. Repeated Play
does not create another state machine, and the final frame stops automatically.

Playback speed controls wall-clock UI pacing, not simulation-model time. The mapping
is:

| Replay speed | Wall-clock interval per stored frame |
|---:|---:|
| 0.5x | 4.00 seconds |
| 1x | 2.00 seconds |
| 2x | 1.00 second |
| 4x | 0.50 seconds |
| 8x | 0.25 seconds |

The default is 4x, so the 32 transitions in the official 33-frame replay take about
16 seconds plus Streamlit rendering time. Every step selects one existing
`ComparativeFrame`; no playback control invokes the Digital Twin. A settings change
stops playback and hides the stale comparison until it is rebuilt.

Both sides always render from the same `ComparativeFrame`, which carries one frame
index, elapsed minute, and simulated clock. The operational view keeps completed and
abandoned counters visible, shows numeric queue and staffing values, and supplements
them with accessible text markers:

- advisor markers say `BUSY` or `FREE` and are capped at ten;
- waiting-contact markers are capped at ten;
- any hidden markers are represented by an explicit additional-count label;
- numeric counts remain authoritative;
- current snapshot pressure remains textual and is not overall-run health.

This remains a replay of completed simulations, not live production telemetry. It
does not modify staffing during a run, synthesize per-frame SLA/occupancy/wait/AHT, or
claim that the assisted decision is optimal.

## Phase 3 storytelling plan

A later Comparative Replay Phase 3 may add evidence-backed decision markers,
CallVerse-assisted narrative, and final visualization polish over the same stored
frames. Phase 2 does not implement recommendation provenance, intervention events,
automatic manager explanations, verdict animation, or export/video recording.
