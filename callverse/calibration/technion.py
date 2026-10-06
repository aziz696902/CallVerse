"""Loader and contact-center calibration for Technion's Anonymous Bank data."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .profiles import (
    ArrivalSlotProfile,
    DataQualityReport,
    SourceMetadata,
    SupportCalibrationProfile,
    kaplan_meier,
    quantile_curve,
    summarize,
)


TECHNION_SOURCE_URL = (
    "https://see-center.iem.technion.ac.il/databases/AnonymousBank/Data/AnonymousBank.pdf"
)
TECHNION_FIELDS = {
    "vru+line",
    "call_id",
    "customer_id",
    "priority",
    "type",
    "date",
    "vru_entry",
    "vru_exit",
    "vru_time",
    "q_start",
    "q_exit",
    "q_time",
    "outcome",
    "ser_start",
    "ser_exit",
    "ser_time",
    "server",
}
ZERO_TIME = {"0:00:00", "00:00:00", ""}


@dataclass(frozen=True)
class _Call:
    call_date: date
    arrival: datetime
    entered_queue: bool
    waiting_minutes: float | None
    service_minutes: float | None
    outcome: str


@dataclass(frozen=True)
class TechnionBuildResult:
    profile: SupportCalibrationProfile
    calls_by_date: tuple[dict[str, object], ...]
    arrival_profile: tuple[dict[str, object], ...]
    service_wait_summary: tuple[dict[str, object], ...]
    abandonment_summary: tuple[dict[str, object], ...]
    arrival_timestamps: tuple[datetime, ...]


def _parse_clock(value: str) -> time:
    return datetime.strptime(value.strip(), "%H:%M:%S").time()


def _combine(call_date: date, value: str, reference: datetime | None = None) -> datetime:
    combined = datetime.combine(call_date, _parse_clock(value))
    if reference is not None and combined < reference:
        if reference - combined > timedelta(hours=12):
            combined += timedelta(days=1)
    return combined


def _integer(row: dict[str, str], field: str) -> int:
    return int(row[field].strip())


def _parse_row(row: dict[str, str]) -> _Call:
    call_date = datetime.strptime(row["date"].strip(), "%y%m%d").date()
    arrival = _combine(call_date, row["vru_entry"])
    vru_exit = _combine(call_date, row["vru_exit"], arrival)
    vru_time = _integer(row, "vru_time")
    queue_time = _integer(row, "q_time")
    service_time = _integer(row, "ser_time")
    if min(vru_time, queue_time, service_time) < 0:
        raise ValueError("negative_duration")
    if vru_exit < arrival or abs((vru_exit - arrival).total_seconds() - vru_time) > 5:
        raise ValueError("impossible_timestamp_ordering")

    outcome = row["outcome"].strip().upper()
    if outcome not in {"AGENT", "HANG", "PHANTOM"}:
        raise ValueError("invalid_outcome")
    if outcome == "PHANTOM":
        raise ValueError("phantom_call")

    entered_queue = row["q_start"].strip() not in ZERO_TIME
    waiting_minutes: float | None = None
    queue_exit: datetime | None = None
    if entered_queue:
        if row["q_exit"].strip() in ZERO_TIME:
            raise ValueError("missing_queue_timestamp")
        queue_start = _combine(call_date, row["q_start"], arrival)
        queue_exit = _combine(call_date, row["q_exit"], queue_start)
        if queue_start < vru_exit - timedelta(seconds=5) or queue_exit < queue_start:
            raise ValueError("impossible_timestamp_ordering")
        if abs((queue_exit - queue_start).total_seconds() - queue_time) > 5:
            raise ValueError("inconsistent_queue_duration")
        waiting_minutes = queue_time / 60
    elif queue_time != 0:
        raise ValueError("queue_duration_without_queue")

    service_minutes: float | None = None
    if outcome == "AGENT":
        if row["ser_start"].strip() in ZERO_TIME or row["ser_exit"].strip() in ZERO_TIME:
            raise ValueError("missing_service_timestamp")
        service_start = _combine(call_date, row["ser_start"], arrival)
        service_exit = _combine(call_date, row["ser_exit"], service_start)
        lower_bound = queue_exit if queue_exit is not None else vru_exit
        if service_start < lower_bound - timedelta(seconds=5) or service_exit < service_start:
            raise ValueError("impossible_timestamp_ordering")
        if service_time <= 0 or abs((service_exit - service_start).total_seconds() - service_time) > 5:
            raise ValueError("invalid_service_duration")
        service_minutes = service_time / 60
    elif service_time != 0:
        raise ValueError("service_duration_for_abandoned_call")

    return _Call(call_date, arrival, entered_queue, waiting_minutes, service_minutes, outcome)


def load_technion(directory: str | Path, *, require_full_year: bool = True) -> TechnionBuildResult:
    directory = Path(directory)
    paths = sorted(directory.glob("*1999.txt"))
    if (require_full_year and len(paths) != 12) or not paths:
        raise FileNotFoundError(f"expected 12 Technion monthly files in {directory}; found {len(paths)}")

    raw_rows = 0
    exclusions: Counter[str] = Counter()
    missingness: Counter[str] = Counter()
    duplicate_rows = 0
    seen: set[tuple[str, str, str]] = set()
    calls: list[_Call] = []

    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not TECHNION_FIELDS.issubset(set(reader.fieldnames or ())):
                missing = sorted(TECHNION_FIELDS - set(reader.fieldnames or ()))
                raise ValueError(f"{path.name} is missing documented fields: {missing}")
            for row in reader:
                raw_rows += 1
                key = (row["date"].strip(), row["vru+line"].strip(), row["call_id"].strip())
                if key in seen:
                    duplicate_rows += 1
                    exclusions["duplicate_call"] += 1
                    continue
                seen.add(key)
                if row["q_start"].strip() in ZERO_TIME:
                    missingness["did_not_enter_queue"] += 1
                if row["ser_start"].strip() in ZERO_TIME:
                    missingness["no_service_timestamp"] += 1
                try:
                    calls.append(_parse_row(row))
                except (KeyError, TypeError, ValueError) as exc:
                    reason = str(exc) if str(exc) else "malformed_row"
                    if reason.startswith("time data") or reason.startswith("invalid literal"):
                        reason = "malformed_value"
                    exclusions[reason] += 1

    usable_rows = len(calls)
    excluded_rows = raw_rows - usable_rows
    if sum(exclusions.values()) != excluded_rows:
        exclusions["other_exclusion"] += excluded_rows - sum(exclusions.values())

    dates = sorted({call.call_date for call in calls})
    by_date = Counter(call.call_date for call in calls)
    daily_counts = [float(by_date[day]) for day in dates]
    slot_counts = Counter(call.arrival.hour * 2 + call.arrival.minute // 30 for call in calls)
    slot_means = [slot_counts[index] / len(dates) for index in range(48)]
    mean_per_slot = sum(slot_means) / 48
    multipliers = [value / mean_per_slot if mean_per_slot else 0.0 for value in slot_means]

    weekday_profiles: dict[str, tuple[float, ...]] = {}
    for weekday in range(7):
        weekday_name = date(2024, 1, 1 + weekday).strftime("%A")
        weekday_dates = {day for day in dates if day.weekday() == weekday}
        counts = Counter(
            call.arrival.hour * 2 + call.arrival.minute // 30
            for call in calls
            if call.call_date in weekday_dates
        )
        means = [counts[index] / len(weekday_dates) if weekday_dates else 0.0 for index in range(48)]
        baseline = sum(means) / 48
        weekday_profiles[weekday_name] = tuple(
            value / baseline if baseline else 0.0 for value in means
        )

    arrival_slots = tuple(
        ArrivalSlotProfile(
            slot=f"{index // 2:02d}:{'30' if index % 2 else '00'}",
            start_minute=index * 30,
            mean_calls=slot_means[index],
            multiplier=multipliers[index],
        )
        for index in range(48)
    )

    service_times = [call.service_minutes for call in calls if call.service_minutes is not None]
    queued = [call for call in calls if call.entered_queue and call.waiting_minutes is not None]
    waiting_times = [call.waiting_minutes for call in queued if call.waiting_minutes is not None]
    abandoned = [call for call in queued if call.outcome == "HANG"]
    survival_observations = [
        (call.waiting_minutes or 0.0, call.outcome == "HANG") for call in queued
    ]
    survival, patience_quantiles = kaplan_meier(
        survival_observations,
        reporting_times=(0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0),
    )

    profile = SupportCalibrationProfile(
        source=SourceMetadata(
            dataset_name="Technion Anonymous Bank Call-Center Data",
            institution="Technion – Israel Institute of Technology",
            source_url=TECHNION_SOURCE_URL,
            period="1999-01-01 through 1999-12-31",
            usage_scope="Generic contact-center arrivals, service, waiting, abandonment, and patience censoring only.",
            license_or_use_statement="Free for use; documentation requests acknowledgement and notification of Avi Mandelbaum.",
        ),
        quality=DataQualityReport(
            raw_rows=raw_rows,
            usable_rows=usable_rows,
            excluded_rows=excluded_rows,
            exclusion_reasons=dict(sorted(exclusions.items())),
            relevant_missingness=dict(sorted(missingness.items())),
            duplicate_rows=duplicate_rows,
            final_grain="one documented call record",
        ),
        observed_days=len(dates),
        daily_call_volume=summarize(daily_counts),
        arrival_slots=arrival_slots,
        weekday_slot_multipliers=weekday_profiles,
        service_time_minutes=summarize(service_times),
        service_time_quantiles=quantile_curve(service_times),
        queued_calls=len(queued),
        waiting_time_minutes=summarize(waiting_times),
        observed_abandonment_rate=len(abandoned) / len(queued) if queued else 0.0,
        abandoned_queued_calls=len(abandoned),
        patience_events=len(abandoned),
        patience_censored=len(queued) - len(abandoned),
        patience_survival=survival,
        patience_quantiles=patience_quantiles,
    )

    calls_by_date = tuple(
        {"date": day.isoformat(), "weekday": day.strftime("%A"), "calls": by_date[day]}
        for day in dates
    )
    arrival_profile = tuple(
        {
            "slot": slot.slot,
            "mean_calls": slot.mean_calls,
            "multiplier": slot.multiplier,
            **{day: weekday_profiles[day][index] for day in weekday_profiles},
        }
        for index, slot in enumerate(arrival_slots)
    )
    service_wait_summary = (
        {"metric": "service_time_minutes", **profile.service_time_minutes.model_dump()},
        {"metric": "waiting_time_minutes", **profile.waiting_time_minutes.model_dump()},
        {"metric": "daily_call_volume", **profile.daily_call_volume.model_dump()},
    )
    abandonment_summary = tuple(point.model_dump() for point in profile.patience_survival)
    return TechnionBuildResult(
        profile,
        calls_by_date,
        arrival_profile,
        service_wait_summary,
        abandonment_summary,
        tuple(call.arrival for call in calls),
    )
