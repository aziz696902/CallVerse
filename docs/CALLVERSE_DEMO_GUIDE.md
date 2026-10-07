# CallVerse V1 Defense and Demo Guide

This guide uses the normal CallVerse application and its real services. Results are not
hardcoded. Allow roughly **7–10 minutes**, excluding an optional live Groq call.

## Start

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
.\.venv\Scripts\streamlit.exe run app.py
```

Select **Manager**. Follow the visible progression:

1. Simulate
2. Observe
3. Forecast
4. Plan workforce
5. Test decision
6. Inspect interaction
7. Evaluate quality

## Three official scenarios

### Demo A — Normal operations

- Preset: `normal_day`
- Seed: `101`
- Mode: calibrated
- Staffing: 8 agents
- Demand multiplier: 1.0
- Verified behavior: 352 contacts, 100% simulated SLA, 0% abandonment, 34.60% occupancy
- Tabs: Scenario Studio, then Twin Monitor
- Story: healthy baseline capacity under this reproducible simulation
- Do not claim: production SLA or that Technion represents delivery customers

### Demo B — Staff shortage

- Preset: `staff_shortage`
- Seed: `404`
- Mode: calibrated
- Staffing: 3 agents
- Demand multiplier: 1.15
- Verified behavior: 394 contacts, 64.73% SLA, 24.62% abandonment, 85.05% occupancy
- Tabs: Scenario Studio, Twin Monitor, Compare Decisions
- Story: constrained capacity creates visible wait/SLA/abandonment degradation
- What-if: change only agents from 3 to 5 and rerun the same seed
- Verified simulated result: SLA 64.73% → 98.41%, abandonment 24.62% → 4.31%,
  average wait 2.11 → 0.20 minutes
- Do not claim: a production A/B test, causal business impact, or guaranteed savings

### Demo C — Perfect storm

- Preset: `perfect_storm`
- Seed: `707`
- Mode: calibrated
- Staffing: 4 agents
- Demand multiplier: 2.5
- Verified behavior: 934 contacts, 27.92% SLA, 53.96% abandonment, 98.42% occupancy
- Tabs: Scenario Studio, Twin Monitor, Forecast, Workforce
- Story: stress scenario motivates proactive demand and capacity planning
- Do not claim: external condition or lateness fields causally generated the demand

## Complete recommended demo path

1. In **Scenario Studio**, run Demo B using its unchanged defaults.
2. Point out generated contacts, wait, SLA, abandonment, occupancy, and AHT.
3. In **Twin Monitor**, show the queue, busy agents, and cumulative outcomes.
4. In **Forecast**, explain the historical Technion contact-demand boundary. Show the
   48 half-hour points, 251.744 total predicted contacts, and 19:00 peak.
5. In **Workforce**, press **BUILD WORKFORCE PLAN**. Explain the calibrated 3.182-minute
   AHT, two-minute SLA threshold, 80% target, 85% occupancy cap, and 10% buffer. The
   example plan uses 1–5 agents, 42.5 agent-hours, and reaches its analytical target
   in 48/48 intervals.
6. In **Compare Decisions**, change Demo B from 3 to 5 agents and run
   **BEFORE VS AFTER**. State: “simulated effect under identical seeded conditions.”
7. In **Interaction Lab**, run **Grounded tracking**. Offline mode proves deterministic
   classifier/tool integration. If Groq quota is available, live mode may be shown and
   must remain labelled live.
8. In **Quality**, show deterministic guardrails. Show six scores only if a real live
   judge completed; otherwise explain the honest unavailable state.
9. Return to **Workforce** and show **PPO Workforce Policy — Experimental**. Explain
   that it trained successfully but was not adopted because it held roughly 17 agents
   and used 408 agent-hours/day.

## Live interaction safety notes

- Known tracking: `CUST-1003`, `ORD-5003`.
- Unknown order: `CUST-1003`, `ORD-9999`; expected safe escalation with no invented status.
- Refund approval: `CUST-1001`, `ORD-5001`; never approve during the demo unless the
  approval workflow itself is being demonstrated.
- A live-provider failure must be shown as unavailable; do not describe offline output
  as live LLM output.

## Measured local timings

| Step | Approximate runtime |
|---|---:|
| Streamlit cold startup | 16.70 s |
| Subsequent UI actions | 0.5–1.3 s |
| Digital Twin run | 0.02 s |
| Forecast load/generation | 0.17 s |
| Workforce plan | <0.01 s |
| Same-seed before/after simulation | 0.02 s |
| Offline Advisor cold start | 5.81 s |
| Offline deterministic Quality | <0.01 s |
| PPO model load and one inference | 3.51 s |
| Live Advisor cases observed | 10–47 s |
| Live Quality judge observed | 3–17 s |

The first Advisor/PPO use is slower because local artifacts initialize. No restart is
needed between tabs.
