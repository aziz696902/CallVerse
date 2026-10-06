"""Leakage-conscious preparation of the Bitext intent subset."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

from .mapping import BITEXT_INTENT_MAPPING, MappingStrength


EXPECTED_COLUMNS = {"flags", "instruction", "category", "intent", "response"}
SPLIT_SEED = 20261006


@dataclass(frozen=True)
class PreparedData:
    rows: tuple[dict[str, str], ...]
    train: tuple[dict[str, str], ...]
    validation: tuple[dict[str, str], ...]
    test: tuple[dict[str, str], ...]
    audit: dict[str, object]


def clean_text(value: str) -> str:
    """Preserve casing/punctuation/misspellings; normalize Unicode and whitespace only."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def leakage_fingerprint(value: str) -> str:
    """Detect case/punctuation-only duplicates without altering model input."""

    return re.sub(r"[^a-z0-9{}]+", " ", value.casefold()).strip()


def _write_rows(path: Path, rows: tuple[dict[str, str], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "label", "source_intent"])
        writer.writeheader()
        writer.writerows(rows)


def prepare_bitext(source_path: str | Path, output_dir: str | Path) -> PreparedData:
    source_path, output_dir = Path(source_path), Path(output_dir)
    raw_rows: list[dict[str, str]] = []
    missing = Counter()
    source_counts = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected Bitext columns: {reader.fieldnames}")
        for raw in reader:
            row = {key: clean_text(value or "") for key, value in raw.items()}
            for field in EXPECTED_COLUMNS:
                if not row[field]:
                    missing[field] += 1
            source_intent = row["intent"]
            source_counts[source_intent] += 1
            if row["instruction"] and len(examples[source_intent]) < 3:
                examples[source_intent].append(row["instruction"])
            raw_rows.append(row)

    selected: list[dict[str, str]] = []
    excluded_missing = 0
    duplicate_pairs = 0
    seen: set[tuple[str, str]] = set()
    for row in raw_rows:
        mapping = BITEXT_INTENT_MAPPING.get(row["intent"])
        if mapping is None or mapping.target is None:
            continue
        if not row["instruction"] or not row["intent"]:
            excluded_missing += 1
            continue
        label = mapping.target.value
        fingerprint = leakage_fingerprint(row["instruction"])
        key = (fingerprint, label)
        if key in seen:
            duplicate_pairs += 1
            continue
        seen.add(key)
        selected.append({"text": row["instruction"], "label": label, "source_intent": row["intent"]})

    labels = [row["label"] for row in selected]
    train_rows, temporary = train_test_split(
        selected, test_size=0.30, random_state=SPLIT_SEED, stratify=labels
    )
    temporary_labels = [row["label"] for row in temporary]
    validation_rows, test_rows = train_test_split(
        temporary, test_size=0.50, random_state=SPLIT_SEED, stratify=temporary_labels
    )
    train, validation, test = tuple(train_rows), tuple(validation_rows), tuple(test_rows)

    def counts(rows: tuple[dict[str, str], ...]) -> dict[str, int]:
        return dict(sorted(Counter(row["label"] for row in rows).items()))

    mapping_audit = {}
    for source_intent in sorted(source_counts):
        mapping = BITEXT_INTENT_MAPPING.get(source_intent)
        mapping_audit[source_intent] = {
            "count": source_counts[source_intent],
            "examples": examples[source_intent],
            "target": mapping.target.value if mapping and mapping.target else None,
            "strength": mapping.strength.value if mapping else MappingStrength.UNSUPPORTED.value,
            "reason": mapping.reason if mapping else "No defensible semantic match to a CallVerse intent.",
        }
    audit: dict[str, object] = {
        "source_rows": len(raw_rows),
        "columns": sorted(EXPECTED_COLUMNS),
        "missing_values": dict(sorted(missing.items())),
        "selected_before_deduplication": sum(
            count for intent, count in source_counts.items() if intent in BITEXT_INTENT_MAPPING
        ),
        "normalized_text_label_duplicates_removed": duplicate_pairs,
        "missing_selected_rows_removed": excluded_missing,
        "usable_rows": len(selected),
        "split_seed": SPLIT_SEED,
        "split_ratio": "70/15/15",
        "split_counts": {"train": counts(train), "validation": counts(validation), "test": counts(test)},
        "source_labels": mapping_audit,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(output_dir / "cleaned_mapped.csv", tuple(selected))
    _write_rows(output_dir / "train.csv", train)
    _write_rows(output_dir / "validation.csv", validation)
    _write_rows(output_dir / "test.csv", test)
    evaluation = output_dir / "evaluation"
    evaluation.mkdir(exist_ok=True)
    (evaluation / "dataset_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (evaluation / "label_mapping.json").write_text(
        json.dumps(mapping_audit, indent=2) + "\n", encoding="utf-8"
    )
    return PreparedData(tuple(selected), train, validation, test, audit)
