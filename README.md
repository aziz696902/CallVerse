# CallVerse

CallVerse is a final-year Data Science research prototype for customer-support and
workforce decision support. It combines a calibrated contact-center Digital Twin,
demand forecasting, analytical workforce planning, synchronized decision replay, an
AI-assisted customer-service workflow, and quality controls in one Streamlit app.

The project is designed to answer operational questions before a manager changes a
real support center. Every dashboard labels simulated, historical, analytical, offline,
and live-LLM evidence separately.

> **Status:** V1 and the Dynamic Workforce research feature are complete. CallVerse is
> a reproducible research prototype, not a production workforce-management system.

## How the application works

```text
Historical contact data
        │
        ├── calibration ──> service, patience and arrival distributions
        │                         │
        │                         ▼
        │                 SimPy Digital Twin
        │                         │
        │                         ├── queues, SLA, abandonment, occupancy
        │                         └── synchronized stored-snapshot replay
        │
        └── LightGBM demand forecast ──> Erlang-C Workforce Manager
                                               │
                                               ▼
                                  48-slot staffing schedule
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        ▼                                             ▼
          Uniform demand-unaware schedule             Forecast-informed schedule
          42.5 agent-hours, pool of 5                 42.5 agent-hours, pool of 5
                        │                                             │
                        └──────── controlled Digital Twin test ───────┘

Customer message ──> intent classifier ──> HelpPilot Advisor
                                           ├── SQLite customer/order tools
                                           ├── Chroma policy retrieval
                                           ├── approval and escalation controls
                                           └── deterministic + optional LLM quality review
```

The primary research comparison keeps the workforce pool, total staffing budget,
seed, realized demand, customer attributes, service draws, and patience draws equal.
Only the time at which staffing capacity is deployed changes.

## Application views

The Streamlit application contains three user areas:

- **Manager Control Room** — scenarios, Digital Twin results, replay, forecast,
  Workforce Intelligence, staffing what-ifs, Interaction Lab, and Quality.
- **Customer Interaction Demo** — customer-facing support conversation and safe action
  handling.
- **Human Approval Queue** — review of sensitive proposed actions such as refunds.

The Manager journey is:

1. **Scenario Studio** configures and runs a calibrated support-center scenario.
2. **Twin Monitor** replays stored queue, advisor, completion, and abandonment states.
3. **Forecast** displays the existing 48-slot LightGBM contact-demand forecast.
4. **Workforce** converts forecast demand into an explainable Erlang-C schedule and
   runs the primary same-resource comparison.
5. **Compare Decisions** tests fixed-capacity what-if decisions under the same seed.
6. **Interaction Lab** exercises the classifier, Advisor, tools, RAG, and escalation.
7. **Quality** applies deterministic safety checks and, when configured, an optional
   structured Groq judge.

## Primary demo: Workforce Intelligence

The jury-facing experiment is **Workforce Intelligence — Same Resource Budget**.

| Controlled condition | Uniform baseline | CallVerse plan |
|---|---:|---:|
| Maximum workforce pool | 5 | 5 |
| Total staffing budget | 42.5 agent-hours | 42.5 agent-hours |
| Simulation horizon | 24 hours | 24 hours |
| Seed | 404 | 404 |
| Realized contacts | 248 | 248 |

The baseline distributes the budget uniformly without accepting forecast or outcome
data. CallVerse allocates the same budget using the existing Forecast-to-Erlang-C plan.
Both schedules are executed by the same Dynamic Twin.

### Verified controlled result

| Metric | Uniform baseline | CallVerse | Difference |
|---|---:|---:|---:|
| Completed contacts | 159 | 228 | +69 |
| SLA | 57.86% | 90.79% | +32.93 pp |
| Abandonment | 35.89% | 8.06% | -27.82 pp |
| Average wait | 3.18 min | 0.54 min | -2.65 min |
| Occupancy | 26.22% | 36.97% | +10.75 pp |
| Final backlog | 0 | 0 | 0 |

Classification: **BETTER ALLOCATION WITH SAME RESOURCE BUDGET**.

All 248 contacts match across the two runs on arrival, intent, persona, patience, and
handling draws. This supports an allocation-timing result inside the calibrated Digital
Twin; it does not prove global optimality or guaranteed production impact.

The app also retains two distinct supporting comparisons:

- **Capacity What-if:** Staff Shortage, seed 404, fixed staffing 3→5.
- **Same-staff reproducibility control:** Staff Shortage, seed 404, 3→3, producing
  identical final KPIs and all 33 identical replay frames.

The older 48.0-hour fixed versus 42.5-hour dynamic test remains documented as secondary
resource-efficiency evidence.

## Quick start

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are recommended.

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
uv sync
Copy-Item .env.example .env
uv run python -m helppilot.seed
uv run streamlit run app.py
```

If the existing virtual environment is already synchronized:

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

The app works offline for the Digital Twin, forecasting, workforce experiments,
classifier, deterministic Advisor demo, and deterministic Quality guardrails. A Groq
key is required only for explicitly labelled live LLM interactions.

Optional secrets belong in the ignored `.env` file:

```dotenv
GROQ_API_KEY=your_real_local_secret
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

Never commit `.env`.

## Recommended two-minute defense flow

1. Introduce the Digital Twin and historical demand boundary.
2. Open **Forecast** and identify the 19:00 predicted peak.
3. Open **Workforce** and select **LOAD RECOMMENDED WORKFORCE DEMO**.
4. Show pool `5`, budget `42.5 vs 42.5`, seed `404`, and the controlled-comparison card.
5. Select **RUN CONTROLLED COMPARISON**, then play at 8x.
6. Pause near 19:00: CallVerse uses more capacity at that moment but used less earlier;
   both policies retain the same daily budget.
7. At 24:00, show exact budget convergence and the operational result.
8. Demonstrate the offline Advisor/RAG path and Quality guardrails.

See the complete [defense guide](docs/CALLVERSE_DEMO_GUIDE.md).

## Reproducibility commands

```powershell
# Full automated suite
.\.venv\Scripts\python.exe -m pytest -q

# Digital Twin deterministic demo
.\.venv\Scripts\python.exe -m callverse.simulation.demo

# Rebuild/evaluate the 24-hour forecast
.\.venv\Scripts\python.exe -m callverse.forecasting.train

# Rebuild Erlang-C versus fixed-Twin validation evidence
.\.venv\Scripts\python.exe -m callverse.workforce.evaluation

# Evaluate the already-trained PPO policy without retraining
.\.venv\Scripts\python.exe -m callverse.rl.evaluation

# Offline classifier and Advisor integration demo
.\.venv\Scripts\python.exe -m callverse.customer_advisor_demo
```

Current verified suite: **260 tests and 15 subtests passing**.

## Implemented components

- calibrated and held-out-validated SimPy operational Digital Twin;
- fixed and immutable 30-minute scheduled-capacity simulation modes;
- non-preemptive staffing reductions with explicit busy-agent overhang;
- seven deterministic scenario presets;
- synchronized 97-frame workforce replay and fixed-capacity comparative replay;
- 48-step LightGBM Poisson historical contact-demand forecast;
- explainable Erlang-C Workforce Manager;
- exact agent-hour accounting and demand-agnostic uniform baseline construction;
- six-class TF-IDF support-intent classifier with safe fallback;
- HelpPilot LangGraph Advisor with SQLite tools, Chroma RAG, approvals, and escalation;
- deterministic Quality guardrails and optional structured Groq evaluation;
- Stable-Baselines3 PPO workforce experiment, evaluated but not adopted operationally.

## Data and evidence boundaries

- **Technion Anonymous Bank Call-Center Data:** contact arrivals, service, waiting, and
  abandonment—not e-commerce customer records.
- **Olist:** delivery and review associations; it does not prove that lateness caused a
  support contact.
- **Bitext:** templated support text for intent classification; its strong held-out score
  must not be generalized to unrestricted production language.
- **Forecast and Erlang-C:** historical prediction and analytical planning evidence, not
  live demand or a guaranteed optimum.
- **Digital Twin:** controlled simulation evidence, not a production A/B test.
- **LLM Quality judge:** optional model output, not human ground truth.

No audio model, weather-causality model, bad-review predictor, or autonomous production
staffing controller is claimed.

## Documentation

- [Architecture and implementation matrix](docs/CALLVERSE_ARCHITECTURE.md)
- [Dynamic Workforce Twin](docs/CALLVERSE_DYNAMIC_WORKFORCE_TWIN.md)
- [Comparative Replay](docs/CALLVERSE_COMPARATIVE_REPLAY.md)
- [Defense and demo guide](docs/CALLVERSE_DEMO_GUIDE.md)
- [Final results](docs/CALLVERSE_FINAL_RESULTS.md)
- [Data sources](docs/DATA_SOURCES.md)
- [PPO experiment](docs/CALLVERSE_PPO_WORKFORCE.md)
- [Storage policy](docs/STORAGE_POLICY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

CallVerse builds on the MIT-licensed
[HelpPilot](https://github.com/poysa213/HelpPilot). Reuse scope, inspected revisions,
and licenses are recorded in `THIRD_PARTY_NOTICES.md`.
