# CallVerse V1

CallVerse is a final-year Data Science research prototype for virtual
support-center decision support. A manager can simulate demand, inspect operational
consequences, forecast upcoming contact volume, build an analytical staffing plan,
test a same-seed what-if decision, and inspect customer-service quality before making
real-world decisions.

**Project implementation status: V1 complete.** CallVerse is a reproducible research
and engineering prototype, not a production workforce or customer-service system.

## Research question and result

Can a calibrated support-center simulation, demand forecast, explainable Erlang-C
baseline, and learned PPO policy provide useful evidence for staffing decisions?

The operational recommendation is the transparent **Erlang-C analytical baseline**.
The experimental PPO policy trained successfully but converged to about 17 constant
agents and 408 agent-hours/day, versus about 118 mean agent-hours for Erlang-C in the
same RL evaluation environment. PPO is therefore **not adopted**.

## Final architecture

```text
Technion operations data + Olist delivery/review data + Bitext support text
                              |
                    calibration / evaluation
                              |
Scenario -> SimPy Digital Twin -> KPIs -> 24h forecast -> Erlang-C plan
                  |                                      |
                  +-------- same-seed what-if -----------+

Customer -> TF-IDF intent classifier -> HelpPilot Advisor
                                      -> SQLite tools / Chroma RAG / approval
                                      -> Quality Analyst

PPO workforce environment -> experimental research result only (not Twin control)
```

See [final architecture and implementation matrix](docs/CALLVERSE_ARCHITECTURE.md)
for evidence boundaries and proposed-versus-implemented components.

## Quick start

Python 3.11 and `uv` are recommended.

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
uv sync
Copy-Item .env.example .env
# Optionally place GROQ_API_KEY in the ignored .env file.
uv run python -m helppilot.seed
uv run streamlit run app.py
```

Open the Manager view for the complete offline demo. Without Groq, the Digital Twin,
forecast, Workforce Manager, PPO research view, offline Interaction Lab, and
deterministic Quality guardrails remain available. Live Advisor and structured Quality
scoring are explicitly unavailable rather than silently replaced with fake LLM output.

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

# Evaluate the already-trained PPO policy (no retraining)
.\.venv\Scripts\python.exe -m callverse.rl.evaluation

# Offline classifier/Advisor integration demo
.\.venv\Scripts\python.exe -m callverse.customer_advisor_demo
```

Optional live configuration belongs only in ignored `.env`:

```dotenv
GROQ_API_KEY=your_real_local_secret
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

LangSmith is optional. Never commit `.env`.

## Implemented modules

- calibrated, held-out-validated SimPy operational Digital Twin;
- seven deterministic scenario presets and same-seed staffing what-if comparison;
- six-class TF-IDF delivery-support intent classifier with safe fallback;
- inherited MIT-licensed HelpPilot LangGraph Advisor, SQLite tools, Chroma RAG,
  grounding review, durable refund approval, and escalation;
- deterministic Quality guardrails plus optional structured Groq six-dimension judge;
- rolling-origin 48-step LightGBM Poisson historical contact-demand forecast;
- calibrated Erlang-C analytical staffing recommendation;
- Stable-Baselines3 PPO workforce experiment, evaluated and rejected operationally;
- one Streamlit application for Customer, Staff, and Manager roles.

## Data and evidence boundaries

- **Technion Anonymous Bank Call-Center Data:** generic contact-center arrivals,
  service, waiting, and abandonment—not e-commerce customer records.
- **Olist:** delivery lateness and review association only; it does not establish that
  lateness caused a support contact.
- **Bitext:** templated/synthetic-like support text used for intent classification;
  perfect held-out TF-IDF scores must not be generalized to production language.

See [data sources](docs/DATA_SOURCES.md),
[final results](docs/CALLVERSE_FINAL_RESULTS.md), and the
[demo guide](docs/CALLVERSE_DEMO_GUIDE.md).

## Major verified results

- Calibration uses 435,785 usable Technion contacts; held-out mean service time was
  3.333 minutes and held-out queued-call abandonment was 23.28%.
- TF-IDF logistic regression achieved 1.000 macro-F1 on the 1,004-row templated test
  split and was selected over BERT-tiny by the predeclared validation rule.
- LightGBM Poisson test MAE was 6.149 contacts/half-hour and RMSE was 10.018.
- The final forecast contains 48 half-hour points and 251.744 predicted contacts.
- The example Erlang-C plan used 42.5 agent-hours and met its analytical target in
  48/48 intervals; this is not a guaranteed production outcome.
- PPO reduced RL-environment wait and abandonment through severe overstaffing and is
  not the operational policy.
- Deterministic Quality safety fixtures pass; live Groq results are integration smoke
  tests, not statistical or human evaluation.

## Limitations

There is no joined late-delivery-to-support-contact dataset, weather causal model,
human evaluation of a large Advisor sample, bad-review prediction model, audio model,
or dynamic staffing inside the frozen Digital Twin. Erlang-C uses simplified M/M/c
assumptions. The PPO environment differs materially from the Twin. A Quality LLM
judge is not human ground truth.

## Documentation and attribution

- [Final results](docs/CALLVERSE_FINAL_RESULTS.md)
- [Defense/demo sequence](docs/CALLVERSE_DEMO_GUIDE.md)
- [Dashboard](docs/CALLVERSE_DASHBOARD.md)
- [PPO experiment](docs/CALLVERSE_PPO_WORKFORCE.md)
- [Storage policy](docs/STORAGE_POLICY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

CallVerse builds on the MIT-licensed
[HelpPilot](https://github.com/poysa213/HelpPilot). Reuse scope, inspected revisions,
and licenses are recorded in `THIRD_PARTY_NOTICES.md`.
