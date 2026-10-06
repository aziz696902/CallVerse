# CallVerse Experimental PPO Workforce Policy

## Status and research question

Phase 11 asks a deliberately narrow question: can a compact PPO policy improve on
fixed staffing and the transparent Phase 10 Erlang-C schedule on held-out historical
demand? This is an experiment, not an automated staffing feature. The result is
negative for deployment: PPO optimizes the internal reward but does not beat Erlang-C
on resource efficiency.

PPO was chosen because staffing is a sequential decision: each interval's capacity
changes the backlog inherited by later intervals. A proven clipped-policy-gradient
implementation supports bounded discrete actions while avoiding a custom RL algorithm.

## Dependency and license gate

CallVerse uses Stable-Baselines3 2.9.0 and Gymnasium 1.4.0 under their MIT licenses.
SB3's wheel is 187.6 kB and Gymnasium's wheel is 476.3 kB. Existing PyTorch 2.13.0
was reused; no second environment, CUDA package, Atari package, or SB3 `extra` bundle
was installed. Exact notices are in `THIRD_PARTY_NOTICES.md`.

## Environment

`WorkforceEnv` is a Gymnasium environment with exactly 48 decisions representing one
day of half-hour intervals. Each action is an absolute staffing level from 1 to 20
agents; SB3 actions 0–19 map to those business values.

The ten-value observation is:

1. normalized carried backlog;
2. normalized current staffing;
3. current forecast contact count;
4. next-interval forecast;
5. mean forecast over the next two hours;
6. previous occupancy;
7. previous interval service level;
8. previous interval abandonment rate;
9. sine of time of day;
10. cosine of time of day.

The demand source is Phase 9's chronological 30-minute series. Training, validation,
and test episodes preserve its existing split boundaries. The forecast signal is the
same slot observed seven days earlier, a Phase 9 seasonal baseline using past data
only. Realized future demand is never present in an observation. Tests explicitly
change future realized demand while holding the observation constant.

The transition model is intentionally lightweight and separate from the authoritative
Digital Twin. It reuses calibrated empirical service-time and patience quantiles,
carries unfinished queue state between intervals, and tracks completions,
abandonments, waits, occupancy, staffing changes, and agent-hours. It batches arrivals
at interval boundaries, which is fast enough for PPO but coarser than continuous-time
SimPy.

## Reward

The default interval reward is:

`2.0 × service level - 0.35 × normalized wait - 6.0 × abandonment rate - 0.20 × normalized staffing - 1.5 × occupancy above 85% - 0.08 × normalized staffing change`

Abandonment has the largest penalty to prevent the policy from appearing efficient by
letting customers leave. The minimum action is one agent, and a high-demand test
confirms that severe understaffing earns less reward than adequate capacity. The
staffing penalty was nevertheless too weak to prevent the learned overstaffing result;
that is retained as evidence rather than tuned away after seeing test outcomes.

## Cross-check against the Digital Twin

Four fixed-demand cases reuse Phase 10's agents and five stochastic seeds. Occupancy
direction is similar, but service quality is not: the RL environment's batch-arrival
approximation produces much longer waits and lower SLA, while its abandonment timing
differs from SimPy. For example, in the normal case occupancy is about 55.4% in both
models, but RL service level is about 19.8% versus 92.6% in the Twin. Therefore the
environment is suitable only for this bounded experiment and cannot replace the Twin.
The full values are in `data/processed/rl/environment_crosscheck.json`.

## Training and reproducibility

The policy is a CPU-only SB3 PPO MLP with two 32-unit hidden layers. Seeds 17, 29, and
43 each ran for 50,000 timesteps. The full run took 131.97 seconds. Selection used the
median validation mean reward, not the best seed; seed 43 was selected. Training used
the Phase 9 train split, selection used only validation, and final comparison used the
held-out test split.

The formal 5,000-step, one-seed smoke run took 6.94 seconds total (4.26 seconds in
learning plus validation). Held-out evaluation took 9.84 seconds. All work ran locally
on CPU; Kaggle, Colab, and GPU were not used. One reward configuration was used. Its
poor staffing-cost sensitivity is part of the reported result; weights were not retuned
after inspecting the held-out test set.

Run locally:

```powershell
Set-Location "C:\Programs\Project_data_science\CallVerse"
.\.venv\Scripts\python.exe -m callverse.rl.train --timesteps 50000
.\.venv\Scripts\python.exe -m callverse.rl.evaluation
.\.venv\Scripts\python.exe -m callverse.rl.validation
```

For Kaggle, upload the repository without `.venv`, select a CPU notebook with network
disabled after dependency setup, install from `uv.lock`, and run the same three Python
modules. The model is tiny and no GPU is required. Preserve seeds, package versions,
split artifacts, and the 50,000-timestep budget when comparing results.

## Held-out comparison

All strategies use the first 12 complete test days and seeds 101, 202, and 303: 36
episodes per strategy with identical episode/seed pairs.

| Strategy | Reward mean ± SD | Abandonment | Service level | Wait (min) | Occupancy | Agent-hours/day | Changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed (5 agents) | 15.46 ± 31.89 | 19.64% | 12.84% | 12.21 | 48.96% | 120.0 | 1.0 |
| Erlang-C | 31.86 ± 21.20 | 2.91% | 14.29% | 10.47 | 45.35% | 118.1 ± 43.9 | 24.3 |
| PPO | 45.82 ± 17.25 | 0.21% | 30.82% | 4.94 | 19.64% | 408.0 | 1.0 |

PPO converged to a nearly constant 17-agent action. Its higher reward, lower wait, and
lower abandonment were purchased with more than three times Erlang-C's mean agent
hours. That is not a useful operational improvement. The correct conclusion is that
this reward/environment combination does **not** establish that RL beats the simple
analytical baseline. Erlang-C remains the recommended explainable planning baseline,
and the calibrated Digital Twin remains the validation authority.

## Artifacts

- `models/ppo_workforce/ppo_policy.zip`: selected representative PPO model;
- `models/ppo_workforce/metadata.json`: versions, seeds, runtime, validation results;
- `data/processed/rl/policy_evaluation.json`: held-out results and trajectories;
- `data/processed/rl/environment_crosscheck.json`: fixed-policy Twin comparison.

Only the selected model is retained. Smoke and unselected seed models are not stored.
