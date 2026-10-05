"""Validated contracts and small statistical helpers for calibration artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean, median, pstdev

from pydantic import Field, field_validator, model_validator

from callverse.domain import DomainModel


DEFAULT_QUANTILE_PROBABILITIES = (
    0.0,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9999,
    1.0,
)


class SourceMetadata(DomainModel):
    dataset_name: str = Field(min_length=1)
    institution: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    period: str = Field(min_length=1)
    usage_scope: str = Field(min_length=1)
    license_or_use_statement: str = Field(min_length=1)


class DataQualityReport(DomainModel):
    raw_rows: int = Field(ge=0)
    usable_rows: int = Field(ge=0)
    excluded_rows: int = Field(ge=0)
    exclusion_reasons: dict[str, int]
    relevant_missingness: dict[str, int]
    duplicate_rows: int = Field(ge=0)
    final_grain: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> DataQualityReport:
        if self.raw_rows != self.usable_rows + self.excluded_rows:
            raise ValueError("raw rows must equal usable rows plus excluded rows")
        if any(count < 0 for count in self.exclusion_reasons.values()):
            raise ValueError("exclusion counts must be non-negative")
        if any(count < 0 for count in self.relevant_missingness.values()):
            raise ValueError("missingness counts must be non-negative")
        return self


class DistributionSummary(DomainModel):
    count: int = Field(ge=0)
    mean: float | None = None
    median: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0)
    minimum: float | None = None
    p25: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_quantile_order(self) -> DistributionSummary:
        if self.count == 0:
            if any(value is not None for key, value in self.__dict__.items() if key != "count"):
                raise ValueError("empty summaries must not contain statistics")
            return self
        ordered = [
            self.minimum,
            self.p25,
            self.median,
            self.p75,
            self.p90,
            self.p95,
            self.p99,
            self.maximum,
        ]
        if any(value is None for value in ordered):
            raise ValueError("non-empty summaries require all quantiles")
        if any(a > b for a, b in zip(ordered, ordered[1:])):  # type: ignore[arg-type]
            raise ValueError("distribution quantiles must be nondecreasing")
        return self


class EmpiricalQuantiles(DomainModel):
    probabilities: tuple[float, ...]
    values: tuple[float, ...]

    @model_validator(mode="after")
    def validate_curve(self) -> EmpiricalQuantiles:
        if len(self.probabilities) != len(self.values) or len(self.values) < 2:
            raise ValueError("quantile probabilities and values must have equal length >= 2")
        if self.probabilities[0] != 0 or self.probabilities[-1] != 1:
            raise ValueError("quantile probabilities must span 0 through 1")
        if any(probability < 0 or probability > 1 for probability in self.probabilities):
            raise ValueError("quantile probabilities must be between 0 and 1")
        if any(a >= b for a, b in zip(self.probabilities, self.probabilities[1:])):
            raise ValueError("quantile probabilities must be strictly increasing")
        if any(value < 0 for value in self.values):
            raise ValueError("duration quantiles must be non-negative")
        if any(a > b for a, b in zip(self.values, self.values[1:])):
            raise ValueError("duration quantiles must be nondecreasing")
        return self


class ArrivalSlotProfile(DomainModel):
    slot: str = Field(pattern=r"^([01]\d|2[0-3]):(00|30)$")
    start_minute: int = Field(ge=0, lt=1440)
    mean_calls: float = Field(ge=0)
    multiplier: float = Field(ge=0)


class SurvivalPoint(DomainModel):
    time_minutes: float = Field(ge=0)
    survival_probability: float = Field(ge=0, le=1)
    at_risk: int = Field(ge=0)
    events: int = Field(ge=0)
    censored: int = Field(ge=0)


class SupportCalibrationProfile(DomainModel):
    profile_version: str = "1.0"
    source: SourceMetadata
    quality: DataQualityReport
    observed_days: int = Field(gt=0)
    daily_call_volume: DistributionSummary
    arrival_slots: tuple[ArrivalSlotProfile, ...]
    weekday_slot_multipliers: dict[str, tuple[float, ...]]
    service_time_minutes: DistributionSummary
    service_time_quantiles: EmpiricalQuantiles
    queued_calls: int = Field(ge=0)
    waiting_time_minutes: DistributionSummary
    observed_abandonment_rate: float = Field(ge=0, le=1)
    abandoned_queued_calls: int = Field(ge=0)
    patience_events: int = Field(ge=0)
    patience_censored: int = Field(ge=0)
    patience_survival: tuple[SurvivalPoint, ...]
    patience_quantiles: EmpiricalQuantiles

    @field_validator("arrival_slots")
    @classmethod
    def validate_slots(cls, slots: tuple[ArrivalSlotProfile, ...]) -> tuple[ArrivalSlotProfile, ...]:
        if len(slots) != 48 or [slot.start_minute for slot in slots] != list(range(0, 1440, 30)):
            raise ValueError("arrival profile must contain ordered 30-minute slots for a full day")
        return slots

    @field_validator("weekday_slot_multipliers")
    @classmethod
    def validate_weekdays(cls, values: dict[str, tuple[float, ...]]) -> dict[str, tuple[float, ...]]:
        expected = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        if set(values) != expected or any(len(profile) != 48 for profile in values.values()):
            raise ValueError("weekday profiles must contain seven named 48-slot profiles")
        if any(multiplier < 0 for profile in values.values() for multiplier in profile):
            raise ValueError("weekday arrival multipliers must be non-negative")
        return values


class ReviewGroupStatistics(DomainModel):
    count: int = Field(ge=0)
    average_score: float | None = Field(default=None, ge=1, le=5)
    low_review_rate: float | None = Field(default=None, ge=0, le=1)
    score_counts: dict[str, int]


class DelayBandStatistics(DomainModel):
    count: int = Field(ge=0)
    average_review_score: float | None = Field(default=None, ge=1, le=5)
    low_review_rate: float | None = Field(default=None, ge=0, le=1)


class DeliveryCalibrationProfile(DomainModel):
    profile_version: str = "1.0"
    source: SourceMetadata
    order_quality: DataQualityReport
    review_quality: DataQualityReport
    eligible_delivered_orders: int = Field(ge=0)
    late_orders: int = Field(ge=0)
    late_delivery_rate: float = Field(ge=0, le=1)
    delivery_delay_days: DistributionSummary
    late_delay_days: DistributionSummary
    early_or_on_time_days: DistributionSummary
    severity_bands: dict[str, DelayBandStatistics]
    reviews_by_delivery_state: dict[str, ReviewGroupStatistics]
    low_review_threshold: int = Field(default=2, ge=1, le=5)
    reviewed_eligible_orders: int = Field(ge=0)


class CalibrationBundle(DomainModel):
    support: SupportCalibrationProfile
    delivery: DeliveryCalibrationProfile


def percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def summarize(values: list[float]) -> DistributionSummary:
    if not values:
        return DistributionSummary(count=0)
    ordered = sorted(values)
    return DistributionSummary(
        count=len(ordered),
        mean=fmean(ordered),
        median=median(ordered),
        standard_deviation=pstdev(ordered),
        minimum=ordered[0],
        p25=percentile(ordered, 0.25),
        p75=percentile(ordered, 0.75),
        p90=percentile(ordered, 0.90),
        p95=percentile(ordered, 0.95),
        p99=percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


def quantile_curve(
    values: list[float],
    probabilities: tuple[float, ...] = DEFAULT_QUANTILE_PROBABILITIES,
) -> EmpiricalQuantiles:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot build empirical quantiles from no values")
    return EmpiricalQuantiles(
        probabilities=probabilities,
        values=tuple(percentile(ordered, probability) for probability in probabilities),
    )


def kaplan_meier(
    observations: list[tuple[float, bool]],
    reporting_times: tuple[float, ...],
) -> tuple[tuple[SurvivalPoint, ...], EmpiricalQuantiles]:
    """Return KM survival points and an inverse-CDF approximation.

    ``event=True`` means observed abandonment; served waits are right-censored.
    The unresolved upper tail is conservatively capped at the largest observed wait.
    """

    if not observations:
        raise ValueError("Kaplan-Meier estimation requires observations")
    grouped: dict[float, list[bool]] = {}
    for duration, event in observations:
        if duration < 0:
            raise ValueError("survival durations must be non-negative")
        grouped.setdefault(duration, []).append(event)

    at_risk = len(observations)
    survival = 1.0
    steps: list[tuple[float, float, int, int, int]] = []
    for duration in sorted(grouped):
        statuses = grouped[duration]
        events = sum(statuses)
        censored = len(statuses) - events
        before = at_risk
        if events and before:
            survival *= 1 - events / before
        steps.append((duration, survival, before, events, censored))
        at_risk -= len(statuses)

    points = []
    for report_time in reporting_times:
        matching = [step for step in steps if step[0] <= report_time]
        if matching:
            _, probability, risk, events, censored = matching[-1]
        else:
            probability, risk, events, censored = 1.0, len(observations), 0, 0
        points.append(
            SurvivalPoint(
                time_minutes=report_time,
                survival_probability=probability,
                at_risk=risk,
                events=events,
                censored=censored,
            )
        )

    probabilities = DEFAULT_QUANTILE_PROBABILITIES
    maximum = max(duration for duration, _ in observations)
    inverse_values = []
    for probability in probabilities:
        target_survival = 1 - probability
        reached = next((duration for duration, surv, *_ in steps if surv <= target_survival), maximum)
        inverse_values.append(reached)
    return tuple(points), EmpiricalQuantiles(probabilities=probabilities, values=tuple(inverse_values))


def load_support_profile(path: str | Path) -> SupportCalibrationProfile:
    return SupportCalibrationProfile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def load_delivery_profile(path: str | Path) -> DeliveryCalibrationProfile:
    return DeliveryCalibrationProfile.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
