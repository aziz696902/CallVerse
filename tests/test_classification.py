from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from callverse.classification.classifier import ModelMetadata, RequestClassifier, load_metadata
from callverse.classification.data import clean_text, prepare_bitext
from callverse.classification.extraction import extract_order_id, infer_urgency
from callverse.classification.mapping import MappingStrength, map_source_intent, BITEXT_INTENT_MAPPING
from callverse.domain import RequestIntent, Urgency


class _Predictor:
    def __init__(self, label: str, confidence: float) -> None:
        self.label, self.confidence = label, confidence

    def predict_proba(self, texts: list[str]) -> tuple[list[str], list[float]]:
        return [self.label] * len(texts), [self.confidence] * len(texts)


def _metadata(threshold: float = 0.7) -> ModelMetadata:
    return ModelMetadata(
        model_name="fixture", model_version="1", model_type="test",
        labels=(RequestIntent.TRACKING, RequestIntent.REFUND),
        confidence_threshold=threshold, dataset_revision="fixture",
        artifact_path="fixture.joblib", artifact_size_bytes=0,
    )


class ClassificationTests(unittest.TestCase):
    def test_label_mapping_is_explicit_and_unsupported_is_not_forced(self) -> None:
        self.assertEqual(map_source_intent("track_order"), RequestIntent.TRACKING)
        self.assertEqual(map_source_intent("get_refund"), RequestIntent.REFUND)
        self.assertEqual(BITEXT_INTENT_MAPPING["get_refund"].strength, MappingStrength.COMBINED)
        self.assertIsNone(map_source_intent("review"))
        self.assertIsNone(map_source_intent("unknown_source_label"))

    def test_cleaning_preserves_case_punctuation_and_misspelling(self) -> None:
        self.assertEqual(clean_text("  My  PARCEL, is ltae?!  "), "My PARCEL, is ltae?!")

    def test_order_id_extraction(self) -> None:
        cases = {
            "Where is ORD-4471?": "ORD-4471",
            "please check order 4471": "ORD-4471",
            "status for #4471": "ORD-4471",
            "no identifier here": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_order_id(text), expected)

    def test_classifier_threshold_and_provisional_urgency(self) -> None:
        accepted = RequestClassifier(_Predictor("tracking", 0.9), _metadata()).classify(
            "URGENT: package ORD-4471 never arrived"
        )
        self.assertFalse(accepted.needs_review)
        self.assertEqual(accepted.extracted_order_id, "ORD-4471")
        self.assertEqual(accepted.urgency, Urgency.HIGH)
        review = RequestClassifier(_Predictor("refund", 0.4), _metadata()).classify("refund please")
        self.assertTrue(review.needs_review)
        self.assertEqual(review.intent, RequestIntent.REFUND)

    def test_metadata_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            path.write_text(json.dumps(_metadata().model_dump(mode="json")), encoding="utf-8")
            self.assertEqual(load_metadata(path), _metadata())

    def test_split_is_deterministic_and_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bitext.csv"
            fields = ["flags", "instruction", "category", "intent", "response"]
            with source.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for intent in ("track_order", "payment_issue"):
                    for index in range(20):
                        writer.writerow({
                            "flags": "B", "instruction": f"{intent} distinct example {index}",
                            "category": "TEST", "intent": intent, "response": "fixture",
                        })
            first = prepare_bitext(source, root / "first")
            second = prepare_bitext(source, root / "second")
            self.assertEqual(first.train, second.train)
            sets = [{row["text"] for row in split} for split in (first.train, first.validation, first.test)]
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_urgency_is_rules_based(self) -> None:
        self.assertEqual(infer_urgency("This is an emergency safety issue"), Urgency.CRITICAL)
        self.assertEqual(infer_urgency("Please answer when possible"), Urgency.NORMAL)


if __name__ == "__main__":
    unittest.main()
