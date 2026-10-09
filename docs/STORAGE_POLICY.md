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
- The final raw Technion, Olist, and Bitext research snapshot and deterministic Bitext
  splits are tracked by explicit project-owner decision for offline reproducibility.
  Runtime databases, vector indexes, checkpoints, secrets, and environments remain
  untracked.

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

## Phase 4 measurement — 2026-10-05

Immediately before Phase 4 the project measured 1,305,049,604 bytes. After full
calibration it measured 1,438,808,673 bytes (about 1.34 GiB), an increase of
133,759,069 bytes. Technion raw files plus
archive occupy 56,714,892 bytes; the two required Olist CSVs plus archive occupy
76,824,164 bytes. Compact processed calibration artifacts total only 73,491 bytes.
The largest new file is the 44,717,580-byte Olist download archive.

The archive hashes were verified and recorded in `docs/DATA_SOURCES.md`.

## Phase 4B cleanup — 2026-10-05

After validation, the verified Technion RAR (10,625,021 bytes), verified Olist ZIP
(44,717,580 bytes), and redundant combined-profile JSON (27,529 bytes) were removed.
Approximately 55.37 MB was recovered. One clean extracted raw copy remains: 46,089,871
bytes of Technion monthly files and 32,106,584 bytes across the two required Olist CSVs.

The final project size is 1,383,493,581 bytes (about 1.29 GiB), including the unchanged
1,302,603,671-byte `.venv`. Raw data totals 78,196,455 bytes and processed calibration
artifacts total 56,998 bytes. The largest remaining non-environment file is
`olist_orders_dataset.csv` at 17,654,914 bytes. Exactly one `.venv` remains, and the
project is far below the 8 GB target.

## Phase 5 measurement — 2026-10-06

Before Phase 5 the project measured 1,383,567,464 bytes. After retaining one ignored
19,202,474-byte Bitext source CSV, ignored deterministic prepared splits, the selected
133,576-byte TF-IDF model, and 35,629 bytes of tracked evaluation evidence, it measured
1,406,017,321 bytes (about 1.31 GiB). Classification processed files total 1,061,328
bytes, including 45,694 bytes of tracked evaluation evidence. All raw data now totals
97,398,929 bytes.

The experimental tiny-BERT checkpoint was removed because it did not outperform the
linear baseline. Its 35,731,594-byte user-level Hugging Face cache and lock directory
were also removed after evaluation. Zero transformer checkpoints remain. The largest
new project file is the ignored Bitext CSV at 19,202,474 bytes. The environment remains
the only `.venv` (1,304,551,556 bytes), and the project remains far below 8 GB.
