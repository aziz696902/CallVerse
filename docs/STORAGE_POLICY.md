# CallVerse Storage Policy

CallVerse should remain reproducible and intentionally lean throughout development.

## Project Rules

- Keep exactly one project-local `.venv`.
- Keep the full local CallVerse project at or below 8 GB where practical.
- Investigate unexpected growth as the project approaches 10 GB.
- Keep tracked Git content ideally at or below 500 MB.
- Do not store local LLM weights unless they are explicitly approved and justified.
- Do not retain unnecessary model checkpoints.
- Do not duplicate datasets inside the project.
- Keep training-heavy artifacts in Kaggle, Colab, or appropriate cloud storage.
- Retain locally only final trained artifacts that are demonstrably useful.
- Require explicit justification before introducing any file larger than about 500 MB.
- Keep generated caches, databases, indexes, logs, and temporary experiment outputs
  untracked unless a small artifact is intentionally required for reproducibility.

## Storage Baseline — 2026-10-05

The maintenance audit measured:

| Area | Size |
|---|---:|
| Entire `CallVerse/` folder | 1,304,711,219 bytes (about 1.22 GiB) |
| `.venv/` | 1,302,603,671 bytes (about 1.21 GiB) |
| `.git/` before the CallVerse checkpoint commit | 366,476 bytes (about 0.35 MiB) |
| Project content excluding `.venv/` and `.git/` | 1,741,072 bytes (about 1.66 MiB) |
| Source/docs/config excluding `.venv/`, `.git/`, `data/`, and caches | 1,087,691 bytes (about 1.04 MiB) |
| Generated `data/` | 450,724 bytes (about 0.43 MiB) |
| Disposable project-level Python caches | 202,657 bytes (about 0.19 MiB) |

Exactly one `.venv` was present. No project-local LLM weights, model checkpoints,
duplicate datasets, large exports, or files above 500 MB were found. The largest files
were compiled runtime dependencies inside `.venv`; the largest was PyTorch's
`torch_cpu.dll` at about 291 MiB.

The `data/` directory contains one HelpPilot SQLite database, one empty LangGraph
checkpoint database, and one Chroma index. These are distinct generated runtime assets,
not duplicate databases, and remain ignored by Git.

Five small `__pycache__` directories were found outside `.venv`. They are disposable
and ignored. Deletion was not necessary for project health; the execution environment
also rejected deletion commands, so this audit recovered zero bytes.
