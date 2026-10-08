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

## Phase 2 animation plan

A later Comparative Replay Phase 2 may iterate over the already prepared frame tuple
to add playback, pause/restart, speed control, and an animated operational view. Phase
1 intentionally provides only the synchronized manual selector; no timers, autoplay,
moving elements, or event-marker interface are implemented.
