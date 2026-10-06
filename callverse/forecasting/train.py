"""Deterministic build and evaluation path for Demand Forecasting V1."""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import lightgbm
import pandas as pd

from .baselines import BASELINE_LAGS, baseline_forecast
from .data import chronological_split, load_demand_series
from .evaluation import rolling_origin_evaluate
from .features import LAGS, ROLLING_WINDOWS
from .model import (
    LIGHTGBM_OBJECTIVES,
    MODEL_VERSION,
    ForecastArtifact,
    fit_lightgbm,
    recursive_lightgbm_forecast,
    save_artifact,
)
from .service import forecast_next_24h, forecast_summary

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TECHNION_DIRECTORY = PROJECT_ROOT / "data/raw/technion_anonymous_bank"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data/processed/forecasting"
ARTIFACT_PATH = PROJECT_ROOT / "models/demand_forecast/selected_model.joblib"


def _baseline_function(model_name: str) -> Callable[[list[float], pd.Timestamp, int], list[float]]:
    return lambda history, _timestamp, horizon: baseline_forecast(history, horizon, model_name)


def _model_function(estimator):
    return lambda history, timestamp, horizon: recursive_lightgbm_forecast(
        estimator, history, timestamp, horizon
    )


def _evaluate_candidates(series: pd.Series, training: pd.Series, period: pd.Series):
    results = {}
    functions = {name: _baseline_function(name) for name in BASELINE_LAGS}
    fitted = {name: fit_lightgbm(training, name) for name in LIGHTGBM_OBJECTIVES}
    functions.update({name: _model_function(model) for name, model in fitted.items()})
    for name, forecast_function in functions.items():
        results[name] = rolling_origin_evaluate(
            series,
            period_start=period.index[0],
            period_end=period.index[-1],
            model_name=name,
            forecast_function=forecast_function,
        )
    return results, fitted


def _series_analysis(series: pd.Series) -> dict[str, object]:
    daily = series.resample("1D").sum()
    zero_calendar_dates = [day.date().isoformat() for day, total in daily.items() if total == 0]
    mean = float(series.mean())
    variance = float(series.var(ddof=0))
    return {
        "date_range": {
            "start": series.index[0].isoformat(),
            "end": series.index[-1].isoformat(),
        },
        "half_hour_slots": len(series),
        "total_contacts": int(series.sum()),
        "zero_count_slots": int((series == 0).sum()),
        "missing_timestamps_after_reindex": int(series.isna().sum()),
        "zero_contact_calendar_days": int((daily == 0).sum()),
        "zero_contact_calendar_dates": zero_calendar_dates,
        "zero_filled_slots_on_those_dates": len(zero_calendar_dates) * 48,
        "mean_contacts_per_slot": mean,
        "variance_contacts_per_slot": variance,
        "variance_to_mean": variance / mean if mean else None,
        "autocorrelation": {
            f"lag_{lag}": float(series.autocorr(lag=lag)) for lag in (1, 2, 48, 336)
        },
        "mean_by_weekday": {
            str(day): float(value)
            for day, value in series.groupby(series.index.dayofweek).mean().items()
        },
        "mean_by_month": {
            str(month): float(value)
            for month, value in series.groupby(series.index.month).mean().items()
        },
        "first_28_days_mean": float(series.iloc[: 28 * 48].mean()),
        "last_28_days_mean": float(series.iloc[-28 * 48 :].mean()),
        "mean_by_half_hour_slot": {
            str(slot): float(value)
            for slot, value in series.groupby(series.index.hour * 2 + series.index.minute // 30)
            .mean()
            .items()
        },
    }


def build_forecast_artifacts() -> dict[str, object]:
    series = load_demand_series(TECHNION_DIRECTORY)
    split = chronological_split(series)

    validation_results, _ = _evaluate_candidates(series, split.train, split.validation)
    selected_name = min(
        validation_results,
        key=lambda name: validation_results[name].overall.mae,
    )

    development = pd.concat([split.train, split.validation])
    test_results, _ = _evaluate_candidates(series, development, split.test)

    final_estimator = (
        fit_lightgbm(series, selected_name) if selected_name in LIGHTGBM_OBJECTIVES else None
    )
    artifact = ForecastArtifact(
        model_name=selected_name,
        model_version=MODEL_VERSION,
        trained_through=series.index[-1].to_pydatetime(),
        estimator=final_estimator,
    )
    save_artifact(artifact, ARTIFACT_PATH)
    forecast = forecast_next_24h(series, ARTIFACT_PATH)
    example = forecast_summary(forecast, series)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    series.to_csv(OUTPUT_DIRECTORY / "demand_30min.csv")
    report = {
        "dataset": _series_analysis(series),
        "splits": {
            "train": [split.train.index[0].isoformat(), split.train.index[-1].isoformat()],
            "validation": [
                split.validation.index[0].isoformat(),
                split.validation.index[-1].isoformat(),
            ],
            "test": [split.test.index[0].isoformat(), split.test.index[-1].isoformat()],
        },
        "strategy": {
            "interval_minutes": 30,
            "horizon_slots": 48,
            "rolling_origin_stride_slots": 48,
            "lags": list(LAGS),
            "rolling_windows": list(ROLLING_WINDOWS),
            "selection_metric": "validation rolling-origin 48-step MAE",
        },
        "validation": {
            name: result.model_dump(mode="json") for name, result in validation_results.items()
        },
        "test": {name: result.model_dump(mode="json") for name, result in test_results.items()},
        "selected_model": selected_name,
        "actual_forecast_example": {
            "forecast_origin": forecast.forecast_origin.isoformat(),
            "first_forecast_timestamp": forecast.points[0].timestamp.isoformat(),
            "last_forecast_timestamp": forecast.points[-1].timestamp.isoformat(),
            **{
                key: value.isoformat() if hasattr(value, "isoformat") else value
                for key, value in example.items()
            },
        },
    }
    (OUTPUT_DIRECTORY / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_name": selected_name,
        "model_version": MODEL_VERSION,
        "trained_through": artifact.trained_through.isoformat(),
        "selection_basis": "lowest validation rolling-origin 48-step MAE",
        "candidate_names": [*BASELINE_LAGS, *LIGHTGBM_OBJECTIVES],
        "lightgbm_version": lightgbm.__version__,
        "selected_lightgbm_parameters": (
            final_estimator.get_params() if final_estimator is not None else None
        ),
        "artifact_path": ARTIFACT_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_bytes": ARTIFACT_PATH.stat().st_size,
        "artifact_sha256": sha256(ARTIFACT_PATH.read_bytes()).hexdigest(),
        "rebuild_command": "uv run python -m callverse.forecasting.train",
    }
    (OUTPUT_DIRECTORY / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    report = build_forecast_artifacts()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
