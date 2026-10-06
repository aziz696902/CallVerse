"""Compact LightGBM models and versioned forecast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor

from .features import feature_row, supervised_features

MODEL_VERSION = "callverse-demand-v1"
LIGHTGBM_OBJECTIVES = {
    "lightgbm_regression": "regression",
    "lightgbm_poisson": "poisson",
}


def build_lightgbm(model_name: str) -> LGBMRegressor:
    if model_name not in LIGHTGBM_OBJECTIVES:
        raise ValueError(f"unknown LightGBM candidate: {model_name}")
    return LGBMRegressor(
        objective=LIGHTGBM_OBJECTIVES[model_name],
        n_estimators=240,
        learning_rate=0.035,
        num_leaves=31,
        max_depth=8,
        min_child_samples=30,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=0.5,
        random_state=20261006,
        deterministic=True,
        force_col_wise=True,
        n_jobs=1,
        verbosity=-1,
    )


def fit_lightgbm(series: pd.Series, model_name: str) -> LGBMRegressor:
    features, target = supervised_features(series)
    model = build_lightgbm(model_name)
    model.fit(features, target)
    return model


def recursive_lightgbm_forecast(
    model: LGBMRegressor,
    history: list[float],
    first_timestamp: datetime | pd.Timestamp,
    horizon: int,
) -> list[float]:
    working = [float(value) for value in history]
    timestamp = pd.Timestamp(first_timestamp)
    predictions: list[float] = []
    for step in range(horizon):
        current = timestamp + pd.Timedelta(minutes=30 * step)
        row = pd.DataFrame([feature_row(current, working)])
        prediction = max(0.0, float(model.predict(row)[0]))
        predictions.append(prediction)
        working.append(prediction)
    return predictions


@dataclass(frozen=True)
class ForecastArtifact:
    model_name: str
    model_version: str
    trained_through: datetime
    estimator: LGBMRegressor | None = None


def save_artifact(artifact: ForecastArtifact, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)


def load_artifact(path: str | Path) -> ForecastArtifact:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"forecast artifact not found: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, ForecastArtifact):
        raise TypeError("forecast artifact has an unexpected type")
    return artifact
