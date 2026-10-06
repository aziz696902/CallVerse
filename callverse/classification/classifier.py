"""Framework-neutral inference boundary for CallVerse request classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from pydantic import Field

from callverse.domain import DomainModel, RequestIntent, Urgency

from .data import clean_text
from .extraction import extract_order_id, infer_urgency


class ModelMetadata(DomainModel):
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    labels: tuple[RequestIntent, ...]
    confidence_threshold: float = Field(ge=0, le=1)
    dataset_revision: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    artifact_size_bytes: int = Field(ge=0)


class ClassificationResult(DomainModel):
    intent: RequestIntent
    confidence: float = Field(ge=0, le=1)
    needs_review: bool
    extracted_order_id: str | None = None
    urgency: Urgency
    urgency_method: str = "rule_based_provisional_v1"
    model_name: str
    model_version: str


class IntentPredictor(Protocol):
    def predict_proba(self, texts: list[str]) -> tuple[list[str], list[float]]: ...


class SklearnPredictor:
    def __init__(self, artifact_path: str | Path) -> None:
        import joblib

        self._pipeline = joblib.load(artifact_path)

    def predict_proba(self, texts: list[str]) -> tuple[list[str], list[float]]:
        probabilities = self._pipeline.predict_proba(texts)
        indices = probabilities.argmax(axis=1)
        labels = [str(self._pipeline.classes_[index]) for index in indices]
        confidences = [float(probabilities[row, index]) for row, index in enumerate(indices)]
        return labels, confidences


class TransformerPredictor:
    def __init__(self, model_dir: str | Path) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, BertTokenizer

        self._torch = torch
        self._tokenizer = BertTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
        self._model.eval()

    def predict_proba(self, texts: list[str]) -> tuple[list[str], list[float]]:
        encoded = self._tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with self._torch.no_grad():
            probabilities = self._torch.softmax(self._model(**encoded).logits, dim=1)
        confidence, indices = probabilities.max(dim=1)
        labels = [self._model.config.id2label[int(index)] for index in indices]
        return labels, [float(value) for value in confidence]


def load_metadata(path: str | Path) -> ModelMetadata:
    return ModelMetadata.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


class RequestClassifier:
    def __init__(self, predictor: IntentPredictor, metadata: ModelMetadata) -> None:
        self._predictor = predictor
        self.metadata = metadata

    @classmethod
    def from_metadata(cls, metadata_path: str | Path) -> RequestClassifier:
        metadata_path = Path(metadata_path)
        metadata = load_metadata(metadata_path)
        artifact_path = Path(metadata.artifact_path)
        if not artifact_path.is_absolute() and not artifact_path.exists():
            for parent in metadata_path.resolve().parents:
                candidate = parent / artifact_path
                if candidate.exists():
                    artifact_path = candidate
                    break
        if metadata.model_type == "tfidf_logistic_regression":
            predictor: IntentPredictor = SklearnPredictor(artifact_path)
        elif metadata.model_type == "transformer":
            predictor = TransformerPredictor(artifact_path)
        else:
            raise ValueError(f"unsupported model type: {metadata.model_type}")
        return cls(predictor, metadata)

    def classify(self, text: str) -> ClassificationResult:
        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("classification text must not be empty")
        labels, confidences = self._predictor.predict_proba([cleaned])
        intent = RequestIntent(labels[0])
        confidence = confidences[0]
        return ClassificationResult(
            intent=intent,
            confidence=confidence,
            needs_review=confidence < self.metadata.confidence_threshold,
            extracted_order_id=extract_order_id(cleaned),
            urgency=infer_urgency(cleaned),
            model_name=self.metadata.model_name,
            model_version=self.metadata.model_version,
        )
