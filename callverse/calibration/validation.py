"""Chronological component validation for the Technion support calibration."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from random import Random
from statistics import fmean, median, pvariance

from callverse.simulation.policies import QuantileMinutes

from .profiles import kaplan_meier, percentile, quantile_curve, summarize
from .technion import TECHNION_FIELDS, _Call, _parse_row


SPLIT_FRACTION = 0.80
SURVIVAL_TIMES = (1.0, 2.0, 5.0, 10.0)
SERVICE_QUANTILES = (0.5, 0.9, 0.95)


@dataclass(frozen=True)
class ValidationResult:
    summary: dict[str, object]
    arrival_rows: tuple[dict[str, object], ...]
    service_rows: tuple[dict[str, object], ...]
    patience_rows: tuple[dict[str, object], ...]


def _read_calls(directory: Path) -> list[_Call]:
    calls: list[_Call] = []
    seen: set[tuple[str, str, str]] = set()
    paths = sorted(directory.glob("*1999.txt"))
    if len(paths) != 12:
        raise FileNotFoundError(f"expected 12 Technion monthly files; found {len(paths)}")
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not TECHNION_FIELDS.issubset(set(reader.fieldnames or ())):
                raise ValueError(f"{path.name} does not match the documented schema")
            for row in reader:
                key = (row["date"].strip(), row["vru+line"].strip(), row["call_id"].strip())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    calls.append(_parse_row(row))
                except (KeyError, TypeError, ValueError):
                    continue
    return calls


def _slot_matrix(calls: list[_Call], dates: list[date]) -> list[list[int]]:
    positions = {day: index for index, day in enumerate(dates)}
    matrix = [[0] * 48 for _ in dates]
    for call in calls:
        if call.call_date in positions:
            matrix[positions[call.call_date]][call.arrival.hour * 2 + call.arrival.minute // 30] += 1
    return matrix


def _slot_statistics(matrix: list[list[int]]) -> tuple[list[float], list[float], list[float]]:
    means, variances, dispersions = [], [], []
    for slot in range(48):
        values = [row[slot] for row in matrix]
        mean = fmean(values)
        variance = pvariance(values)
        means.append(mean)
        variances.append(variance)
        dispersions.append(variance / mean if mean else 0.0)
    return means, variances, dispersions


def _normalized(values: list[float]) -> list[float]:
    baseline = fmean(values)
    return [value / baseline if baseline else 0.0 for value in values]


def _ks(first: list[float], second: list[float]) -> float:
    left, right = sorted(first), sorted(second)
    i = j = 0
    distance = 0.0
    while i < len(left) or j < len(right):
        if j == len(right) or (i < len(left) and left[i] <= right[j]):
            value = left[i]
        else:
            value = right[j]
        while i < len(left) and left[i] <= value:
            i += 1
        while j < len(right) and right[j] <= value:
            j += 1
        distance = max(distance, abs(i / len(left) - j / len(right)))
    return distance


def _wasserstein(first: list[float], second: list[float]) -> float:
    left, right = sorted(first), sorted(second)
    probabilities = [(index + 0.5) / max(len(left), len(right)) for index in range(max(len(left), len(right)))]
    return fmean(
        abs(percentile(left, probability) - percentile(right, probability))
        for probability in probabilities
    )


def _service_candidates(calibration: list[float], size: int) -> dict[str, list[float]]:
    rng = Random(20261005)
    mean = fmean(calibration)
    variance = pvariance(calibration)
    logs = [math.log(value) for value in calibration]
    log_mean, log_sigma = fmean(logs), math.sqrt(pvariance(logs))
    gamma_shape = mean * mean / variance
    gamma_scale = variance / mean
    compact = quantile_curve(calibration)
    sampler = QuantileMinutes(compact.probabilities, compact.values)
    return {
        "phase3_triangular": [rng.triangular(3.0, 10.0, 6.0) for _ in range(size)],
        "exponential": [rng.expovariate(1 / mean) for _ in range(size)],
        "lognormal": [rng.lognormvariate(log_mean, log_sigma) for _ in range(size)],
        "gamma_moments": [rng.gammavariate(gamma_shape, gamma_scale) for _ in range(size)],
        "empirical_quantiles": [sampler.sample(rng) for _ in range(size)],
    }


def validate_technion(directory: str | Path) -> ValidationResult:
    calls = _read_calls(Path(directory))
    dates = sorted({call.call_date for call in calls})
    split_index = math.floor(len(dates) * SPLIT_FRACTION)
    calibration_dates, validation_dates = dates[:split_index], dates[split_index:]
    split_date = validation_dates[0]
    calibration = [call for call in calls if call.call_date < split_date]
    validation = [call for call in calls if call.call_date >= split_date]

    calibration_matrix = _slot_matrix(calibration, calibration_dates)
    validation_matrix = _slot_matrix(validation, validation_dates)
    cal_means, cal_variances, cal_dispersion = _slot_statistics(calibration_matrix)
    val_means, val_variances, val_dispersion = _slot_statistics(validation_matrix)
    cal_shape, val_shape = _normalized(cal_means), _normalized(val_means)
    arrival_rows = tuple(
        {
            "slot": f"{slot // 2:02d}:{'30' if slot % 2 else '00'}",
            "calibration_mean": cal_means[slot],
            "validation_mean": val_means[slot],
            "calibration_variance": cal_variances[slot],
            "validation_variance": val_variances[slot],
            "calibration_dispersion": cal_dispersion[slot],
            "validation_dispersion": val_dispersion[slot],
            "calibration_normalized": cal_shape[slot],
            "validation_normalized": val_shape[slot],
        }
        for slot in range(48)
    )
    flat_calibration = [count for row in calibration_matrix for count in row]
    flat_validation = [count for row in validation_matrix for count in row]

    cal_service = [call.service_minutes for call in calibration if call.service_minutes is not None]
    val_service = [call.service_minutes for call in validation if call.service_minutes is not None]
    val_ordered = sorted(val_service)
    service_rows = []
    for name, samples in _service_candidates(cal_service, len(val_service)).items():
        ordered = sorted(samples)
        errors = {
            f"p{int(probability * 100)}_error": percentile(ordered, probability) - percentile(val_ordered, probability)
            for probability in SERVICE_QUANTILES
        }
        service_rows.append({
            "representation": name,
            "ks_statistic": _ks(val_service, samples),
            "wasserstein_minutes": _wasserstein(val_service, samples),
            **errors,
        })

    def patience(calls_for_period: list[_Call]):
        queued = [call for call in calls_for_period if call.entered_queue and call.waiting_minutes is not None]
        observations = [(call.waiting_minutes or 0.0, call.outcome == "HANG") for call in queued]
        points, _ = kaplan_meier(observations, SURVIVAL_TIMES)
        return queued, points

    cal_queued, cal_survival = patience(calibration)
    val_queued, val_survival = patience(validation)
    patience_rows = tuple(
        {
            "time_minutes": time,
            "calibration_survival": cal_point.survival_probability,
            "validation_survival": val_point.survival_probability,
            "absolute_difference": abs(cal_point.survival_probability - val_point.survival_probability),
        }
        for time, cal_point, val_point in zip(SURVIVAL_TIMES, cal_survival, val_survival)
    )
    cal_abandoned = sum(call.outcome == "HANG" for call in cal_queued)
    val_abandoned = sum(call.outcome == "HANG" for call in val_queued)
    summary: dict[str, object] = {
        "split": {
            "method": "first 80% of observed dates for calibration; final 20% for held-out validation",
            "calibration_start": calibration_dates[0].isoformat(),
            "calibration_end": calibration_dates[-1].isoformat(),
            "calibration_dates": len(calibration_dates),
            "validation_start": validation_dates[0].isoformat(),
            "validation_end": validation_dates[-1].isoformat(),
            "validation_dates": len(validation_dates),
            "calibration_calls": len(calibration),
            "validation_calls": len(validation),
        },
        "arrival": {
            "calibration_flat_mean": fmean(flat_calibration),
            "calibration_flat_dispersion": pvariance(flat_calibration) / fmean(flat_calibration),
            "validation_flat_mean": fmean(flat_validation),
            "validation_flat_dispersion": pvariance(flat_validation) / fmean(flat_validation),
            "calibration_median_within_slot_dispersion": median(cal_dispersion),
            "validation_median_within_slot_dispersion": median(val_dispersion),
            "normalized_slot_mae": fmean(abs(left - right) for left, right in zip(cal_shape, val_shape)),
            "raw_slot_mean_mae": fmean(abs(left - right) for left, right in zip(cal_means, val_means)),
        },
        "service": {
            "calibration_count": len(cal_service),
            "validation_count": len(val_service),
            "calibration_summary": summarize(cal_service).model_dump(mode="json"),
            "validation_summary": summarize(val_service).model_dump(mode="json"),
            "selected_representation": "empirical_quantiles",
        },
        "patience": {
            "calibration_queued": len(cal_queued),
            "validation_queued": len(val_queued),
            "calibration_events": cal_abandoned,
            "validation_events": val_abandoned,
            "calibration_censored": len(cal_queued) - cal_abandoned,
            "validation_censored": len(val_queued) - val_abandoned,
            "calibration_abandonment_rate": cal_abandoned / len(cal_queued),
            "validation_abandonment_rate": val_abandoned / len(val_queued),
        },
    }
    return ValidationResult(summary, arrival_rows, tuple(service_rows), patience_rows)


def write_validation(result: ValidationResult, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "heldout_validation.json").write_text(
        json.dumps(result.summary, indent=2) + "\n", encoding="utf-8"
    )
    for name, rows in (
        ("arrival_validation.csv", result.arrival_rows),
        ("service_fit_validation.csv", result.service_rows),
        ("patience_validation.csv", result.patience_rows),
    ):
        with (output / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
