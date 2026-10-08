# CallVerse V1 Defense and Demo Guide

This guide uses the normal CallVerse application and its real services. Results are not
hardcoded. Allow roughly **7–10 minutes**, excluding an optional live Groq call.

## Start

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
.\.venv\Scripts\streamlit.exe run app.py
```

The app opens on **Manager Control Room**, the primary demo surface. Follow the visible
progression:

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
- Tabs: Scenario Studio, Twin Monitor, Forecast, Workforce, Compare Decisions
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
3. In **Twin Monitor**, move the replay slider. Distinguish **current snapshot
   pressure** from the completed/abandoned counters accumulated earlier in the run.
4. Still in **Twin Monitor**, use **PREPARE PRIMARY TEACHING DEMO**, then **BUILD
   COMPARISON**. Play or scrub the matched Staff Shortage 3-to-5 replay. Explain the
   tested decision card, deterministic event markers, current-versus-cumulative
   evidence, queue trajectory, and final manager summary. State: "controlled simulation
   comparison, not a production A/B test."
5. In **Forecast**, explain the historical Technion contact-demand boundary. Show the
   48 half-hour points, 251.744 total predicted contacts, and 19:00 peak.
6. In **Workforce**, press **BUILD WORKFORCE PLAN**. Explain the calibrated 3.182-minute
   AHT, two-minute SLA threshold, 80% target, 85% occupancy cap, and 10% buffer. The
   example plan uses 1–5 agents, 42.5 agent-hours, and reaches its analytical target
   in 48/48 intervals.
   Then run **FAIR WORKFORCE COMPARISON**: the fixed two-advisor baseline uses 48.0
   agent-hours, while the predefined Forecast-to-Erlang-C schedule uses 42.5. Under
   identical realized demand, show the actual simulated SLA, abandonment, and wait
   results. Call this **workforce intelligence**, not proof of a production optimum.
7. In **Compare Decisions**, press **PREPARE RECOMMENDED DECISION TEST**, then
   **RUN COMPARISON**. State: “simulated effect under identical seeded conditions.”
8. In **Interaction Lab**, run **Grounded tracking**. Offline mode proves deterministic
   classifier/tool integration. If Groq quota is available, live mode may be shown and
   must remain labelled live.
9. In **Quality**, show deterministic guardrails. Show six scores only if a real live
   judge completed; otherwise explain the honest unavailable state.
10. Return to **Workforce** and show **EXPERIMENTAL PPO POLICY — NOT ADOPTED**. Explain
   that it trained successfully but was not adopted because it held roughly 17 agents
   and used 408 agent-hours/day.

## Optional scalability demonstration

After the primary route, select **PREPARE LARGE CENTER STRESS TEST** in Comparative
Replay and build the predefined seed-404, 5x configuration: demand multiplier 5.75 and
15 baseline to 25 assisted advisors. Use it only to show that the same simulator,
synchronized playback, storytelling, and capped visuals remain practical at a larger
simulated scale. State that it is a simulated scalability demonstration and is not
separately calibrated to a real large call center.

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
| Streamlit cold startup | 7.36 s |
| Digital Twin run | 0.02 s direct / 0.37 s UI |
| Replay interaction | <0.01 s frame / 0.34 s UI rerender |
| Forecast load/generation | 0.11 s |
| Workforce plan | <0.01 s direct / 0.44 s UI |
| Same-seed before/after simulation | 0.02 s direct / 0.46 s UI |
| Large-center comparison build | 0.293 s direct / 0.641 s UI |
| Large-center stored-frame access / scrub | <0.001 ms direct / 0.308 s UI |
| Offline Advisor cold start | 1.33 s |
| Offline deterministic Quality | <0.01 s |
| PPO model load and one inference | 1.69 s |

Measured locally on 2026-10-08. The first Advisor/PPO use is slower because local
artifacts initialize. No restart is needed between tabs. Live Groq latency was not
retested because Phase 4 did not change provider behavior.

## Evidence language for the defense

- **SIMULATED:** Scenario Studio, Twin Monitor, and same-seed Compare Decisions.
- **HISTORICAL ML FORECAST:** LightGBM Poisson contact-demand forecast; not live
  weather and not customer satisfaction.
- **ANALYTICAL ERLANG-C BASELINE:** a staffing calculation from forecast demand; not
  an optimality claim or guarantee.
- **OFFLINE DETERMINISTIC / LIVE LLM-ASSISTED:** Interaction Lab always names its
  execution mode. Structured tools provide customer/order facts; RAG provides policy
  and procedure context.
- **DETERMINISTIC GUARDRAILS / OPTIONAL LLM JUDGE:** hard safety and compliance flags
  remain active without an LLM; nuanced scores are optional and are not human ground
  truth.

Customer Interaction Demo and Human Approval Queue are supporting views. They
demonstrate, respectively, the deeper Advisor pipeline and human review of sensitive
actions; neither replaces the primary manager journey.
