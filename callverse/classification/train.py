"""Reproducible baselines and compact transformer experiment for Phase 5."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

from .data import PreparedData, prepare_bitext


LABELS = ("address_change", "cancel_order", "complaint", "payment_issue", "refund", "tracking")
BASE_MODEL = "prajjwal1/bert-tiny"
DATASET_REVISION = "430d1a89bd93bd1fa23c16f29dd53e73f0087443"
TRAINING_SEED = 20261006


def _xy(rows: tuple[dict[str, str], ...]) -> tuple[list[str], list[str]]:
    return [row["text"] for row in rows], [row["label"] for row in rows]


def _metrics(actual: list[str], predicted: list[str]) -> dict[str, object]:
    report = classification_report(
        actual, predicted, labels=list(LABELS), output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(LABELS)).tolist(),
    }


def _draw_confusion(matrix: list[list[int]], path: Path) -> None:
    size, margin = 760, 170
    cell = (size - margin) // len(LABELS)
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    maximum = max(max(row) for row in matrix) or 1
    for row, actual in enumerate(LABELS):
        draw.text((5, margin + row * cell + cell // 2), actual, fill="black")
        draw.text((margin + row * cell + 4, 25), actual, fill="black")
        for column, value in enumerate(matrix[row]):
            intensity = int(245 - 180 * value / maximum)
            box = (
                margin + column * cell,
                margin + row * cell,
                margin + (column + 1) * cell,
                margin + (row + 1) * cell,
            )
            draw.rectangle(box, fill=(intensity, intensity, 255), outline="gray")
            draw.text((box[0] + cell // 2 - 8, box[1] + cell // 2 - 5), str(value), fill="black")
    draw.text((margin, 5), "Predicted →", fill="black")
    draw.text((5, margin - 25), "Actual ↓", fill="black")
    image.save(path)


def _train_tfidf(data: PreparedData, model_dir: Path):
    _, train_labels = _xy(data.train)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=50_000, sublinear_tf=True)),
        ("classifier", LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=TRAINING_SEED)),
    ])
    pipeline.fit(train_text, train_labels)
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "tfidf_logistic.joblib"
    joblib.dump(pipeline, path, compress=3)
    return pipeline, path


def _transformer_predictions(model, tokenizer, texts: list[str], batch_size: int = 64):
    import torch

    model.eval()
    probabilities = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size], padding=True, truncation=True,
            max_length=64, return_tensors="pt",
        )
        with torch.no_grad():
            probabilities.append(torch.softmax(model(**encoded).logits, dim=1).cpu().numpy())
    matrix = np.concatenate(probabilities)
    indices = matrix.argmax(axis=1)
    return [model.config.id2label[int(index)] for index in indices], matrix.max(axis=1).tolist()


def _train_transformer(data: PreparedData, output_dir: Path):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    config = BertConfig.from_pretrained(
        BASE_MODEL, num_labels=len(LABELS), label2id=label2id, id2label=id2label
    )
    model = BertForSequenceClassification.from_pretrained(
        BASE_MODEL, config=config, ignore_mismatched_sizes=True
    )

    class TextDataset(Dataset):
        def __init__(self, rows: tuple[dict[str, str], ...]) -> None:
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            return row["text"], label2id[row["label"]]

    def collate(batch):
        texts, labels = zip(*batch)
        encoded = tokenizer(list(texts), padding=True, truncation=True, max_length=64, return_tensors="pt")
        encoded["labels"] = torch.tensor(labels)
        return encoded

    generator = torch.Generator().manual_seed(TRAINING_SEED)
    loader = DataLoader(TextDataset(data.train), batch_size=32, shuffle=True, collate_fn=collate, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    validation_text, validation_labels = _xy(data.validation)
    best_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, 5):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad()
            result = model(**batch)
            result.loss.backward()
            optimizer.step()
            losses.append(float(result.loss.detach()))
        predicted, _ = _transformer_predictions(model, tokenizer, validation_text)
        macro_f1 = f1_score(validation_labels, predicted, labels=list(LABELS), average="macro", zero_division=0)
        history.append({"epoch": epoch, "mean_training_loss": float(np.mean(losses)), "validation_macro_f1": macro_f1})
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, history


def _tune_threshold(actual: list[str], predicted: list[str], confidence: list[float]) -> dict[str, float]:
    candidates = [value / 100 for value in range(50, 96, 5)]
    rows = []
    for threshold in candidates:
        accepted = [index for index, value in enumerate(confidence) if value >= threshold]
        coverage = len(accepted) / len(actual)
        if not accepted:
            continue
        accepted_actual = [actual[index] for index in accepted]
        accepted_predicted = [predicted[index] for index in accepted]
        rows.append({
            "threshold": threshold,
            "coverage": coverage,
            "review_rate": 1 - coverage,
            "accepted_accuracy": accuracy_score(accepted_actual, accepted_predicted),
            "accepted_macro_f1": f1_score(
                accepted_actual, accepted_predicted, labels=list(LABELS), average="macro", zero_division=0
            ),
        })
    eligible = [row for row in rows if row["coverage"] >= 0.80]
    return max(eligible or rows, key=lambda row: (row["accepted_macro_f1"], row["coverage"]))


def train_and_evaluate(data: PreparedData, model_dir: str | Path, evaluation_dir: str | Path) -> dict[str, object]:
    model_dir, evaluation_dir = Path(model_dir), Path(evaluation_dir)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    train_text, train_labels = _xy(data.train)
    validation_text, validation_labels = _xy(data.validation)
    test_text, test_labels = _xy(data.test)

    majority = Counter(train_labels).most_common(1)[0][0]
    majority_metrics = _metrics(test_labels, [majority] * len(test_labels))

    tfidf, tfidf_path = _train_tfidf(data, model_dir)
    tfidf_validation_probabilities = tfidf.predict_proba(validation_text)
    tfidf_validation_indices = tfidf_validation_probabilities.argmax(axis=1)
    tfidf_validation_predictions = [str(tfidf.classes_[index]) for index in tfidf_validation_indices]
    tfidf_test_probabilities = tfidf.predict_proba(test_text)
    tfidf_test_indices = tfidf_test_probabilities.argmax(axis=1)
    tfidf_test_predictions = [str(tfidf.classes_[index]) for index in tfidf_test_indices]
    tfidf_metrics = _metrics(test_labels, tfidf_test_predictions)

    transformer_dir = model_dir / "transformer_candidate"
    transformer, tokenizer, history = _train_transformer(data, transformer_dir)
    transformer_validation_predictions, transformer_validation_confidence = _transformer_predictions(
        transformer, tokenizer, validation_text
    )
    transformer_test_predictions, transformer_test_confidence = _transformer_predictions(
        transformer, tokenizer, test_text
    )
    transformer_metrics = _metrics(test_labels, transformer_test_predictions)
    with (evaluation_dir / "transformer_errors.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual", "predicted", "confidence", "text"])
        writer.writeheader()
        for actual, predicted, confidence, text in zip(
            test_labels, transformer_test_predictions, transformer_test_confidence, test_text
        ):
            if actual != predicted:
                writer.writerow({"actual": actual, "predicted": predicted, "confidence": confidence, "text": text})

    tfidf_validation_metrics = _metrics(validation_labels, tfidf_validation_predictions)
    transformer_validation_metrics = _metrics(validation_labels, transformer_validation_predictions)
    selected = "transformer" if transformer_validation_metrics["macro_f1"] > tfidf_validation_metrics["macro_f1"] + 0.005 else "tfidf_logistic"
    if selected == "transformer":
        validation_predictions = transformer_validation_predictions
        validation_confidence = transformer_validation_confidence
        test_predictions = transformer_test_predictions
        test_confidence = transformer_test_confidence
        final_metrics = transformer_metrics
        final_artifact = transformer_dir
    else:
        validation_predictions = tfidf_validation_predictions
        validation_confidence = tfidf_validation_probabilities.max(axis=1).tolist()
        test_predictions = tfidf_test_predictions
        test_confidence = tfidf_test_probabilities.max(axis=1).tolist()
        final_metrics = tfidf_metrics
        final_artifact = tfidf_path
        shutil.rmtree(transformer_dir)

    threshold = _tune_threshold(validation_labels, validation_predictions, validation_confidence)
    accepted = [index for index, value in enumerate(test_confidence) if value >= threshold["threshold"]]
    test_threshold = {
        "threshold": threshold["threshold"],
        "coverage": len(accepted) / len(test_labels),
        "review_rate": 1 - len(accepted) / len(test_labels),
        "accepted_accuracy": accuracy_score([test_labels[i] for i in accepted], [test_predictions[i] for i in accepted]),
        "accepted_macro_f1": f1_score(
            [test_labels[i] for i in accepted], [test_predictions[i] for i in accepted],
            labels=list(LABELS), average="macro", zero_division=0,
        ),
    }
    metadata = {
        "model_name": selected,
        "model_version": "phase5-v1",
        "model_type": "transformer" if selected == "transformer" else "tfidf_logistic_regression",
        "labels": list(LABELS),
        "confidence_threshold": threshold["threshold"],
        "dataset_revision": DATASET_REVISION,
        "artifact_path": str(final_artifact).replace("\\", "/"),
        "artifact_size_bytes": (
            sum(path.stat().st_size for path in final_artifact.rglob("*") if path.is_file())
            if final_artifact.is_dir() else final_artifact.stat().st_size
        ),
    }
    results: dict[str, object] = {
        "dataset_revision": DATASET_REVISION,
        "split_seed": TRAINING_SEED,
        "test_rows": len(test_labels),
        "majority_class": majority,
        "majority": majority_metrics,
        "tfidf_logistic": tfidf_metrics,
        "transformer": transformer_metrics,
        "transformer_base_model": BASE_MODEL,
        "transformer_training_history": history,
        "selection_rule": "Transformer selected only if validation macro-F1 exceeds TF-IDF by more than 0.005.",
        "selected_model": selected,
        "validation_threshold_selection": threshold,
        "test_confidence_results": test_threshold,
        "final": final_metrics,
    }
    (evaluation_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (evaluation_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _draw_confusion(final_metrics["confusion_matrix"], evaluation_dir / "confusion_matrix.png")
    with (evaluation_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["actual", "predicted", "confidence", "needs_review", "text"])
        writer.writeheader()
        for actual, predicted, confidence, text in zip(
            test_labels, test_predictions, test_confidence, test_text
        ):
            if actual != predicted:
                writer.writerow({
                    "actual": actual, "predicted": predicted, "confidence": confidence,
                    "needs_review": confidence < threshold["threshold"], "text": text,
                })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/classification"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/intent_classifier"))
    args = parser.parse_args()
    data = prepare_bitext(args.source, args.processed_dir)
    result = train_and_evaluate(data, args.model_dir, args.processed_dir / "evaluation")
    print(json.dumps({name: result[name] for name in ("selected_model", "majority", "tfidf_logistic", "transformer", "test_confidence_results")}, indent=2))


if __name__ == "__main__":
    main()
