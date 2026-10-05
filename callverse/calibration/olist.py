"""Order-grain delivery calibration from the official Olist data files."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .profiles import (
    DataQualityReport,
    DelayBandStatistics,
    DeliveryCalibrationProfile,
    ReviewGroupStatistics,
    SourceMetadata,
    summarize,
)


OLIST_SOURCE_URL = "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
ORDER_FIELDS = {
    "order_id", "order_status", "order_purchase_timestamp",
    "order_delivered_customer_date", "order_estimated_delivery_date",
}
REVIEW_FIELDS = {
    "review_id", "order_id", "review_score", "review_creation_date",
    "review_answer_timestamp",
}
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class _Order:
    order_id: str
    delay_days: float


@dataclass(frozen=True)
class _Review:
    order_id: str
    review_id: str
    score: int
    creation: datetime
    answer: datetime


@dataclass(frozen=True)
class OlistBuildResult:
    profile: DeliveryCalibrationProfile
    lateness_summary: tuple[dict[str, object], ...]
    review_summary: tuple[dict[str, object], ...]


def _timestamp(value: str) -> datetime:
    return datetime.strptime(value.strip(), TIMESTAMP_FORMAT)


def _severity(delay_days: float) -> str:
    if delay_days <= 0:
        return "on_time_or_early"
    if delay_days <= 2:
        return "slightly_late"
    if delay_days <= 7:
        return "moderately_late"
    return "severely_late"


def _quality(raw: int, usable: int, exclusions: Counter[str], missing: Counter[str], duplicates: int, grain: str) -> DataQualityReport:
    excluded = raw - usable
    difference = excluded - sum(exclusions.values())
    if difference:
        exclusions["other_exclusion"] += difference
    return DataQualityReport(
        raw_rows=raw,
        usable_rows=usable,
        excluded_rows=excluded,
        exclusion_reasons=dict(sorted(exclusions.items())),
        relevant_missingness=dict(sorted(missing.items())),
        duplicate_rows=duplicates,
        final_grain=grain,
    )


def load_olist(orders_path: str | Path, reviews_path: str | Path) -> OlistBuildResult:
    """Load only orders and reviews, retaining exactly one analytical row per order."""

    orders: dict[str, _Order] = {}
    order_raw = order_duplicates = 0
    order_exclusions: Counter[str] = Counter()
    order_missing: Counter[str] = Counter()
    with Path(orders_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = ORDER_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"orders file is missing fields: {sorted(missing_fields)}")
        for row in reader:
            order_raw += 1
            order_id = row["order_id"].strip()
            if order_id in orders:
                order_duplicates += 1
                order_exclusions["duplicate_order_id"] += 1
                continue
            if row["order_status"].strip().lower() != "delivered":
                order_exclusions["not_delivered"] += 1
                continue
            required = ("order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date")
            absent = [field for field in required if not row[field].strip()]
            if absent:
                for field in absent:
                    order_missing[field] += 1
                order_exclusions["missing_required_timestamp"] += 1
                continue
            try:
                purchase = _timestamp(row["order_purchase_timestamp"])
                actual = _timestamp(row["order_delivered_customer_date"])
                estimated = _timestamp(row["order_estimated_delivery_date"])
            except ValueError:
                order_exclusions["malformed_timestamp"] += 1
                continue
            if actual < purchase:
                order_exclusions["delivery_before_purchase"] += 1
                continue
            orders[order_id] = _Order(order_id, (actual - estimated).total_seconds() / 86_400)

    reviews: dict[str, _Review] = {}
    review_raw = review_duplicates = 0
    review_exclusions: Counter[str] = Counter()
    review_missing: Counter[str] = Counter()
    with Path(reviews_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_fields = REVIEW_FIELDS - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(f"reviews file is missing fields: {sorted(missing_fields)}")
        valid_review_rows = 0
        for row in reader:
            review_raw += 1
            required = ("review_id", "order_id", "review_score", "review_creation_date", "review_answer_timestamp")
            absent = [field for field in required if not row[field].strip()]
            if absent:
                for field in absent:
                    review_missing[field] += 1
                review_exclusions["missing_required_value"] += 1
                continue
            try:
                score = int(row["review_score"])
                creation = _timestamp(row["review_creation_date"])
                answer = _timestamp(row["review_answer_timestamp"])
                if score < 1 or score > 5:
                    raise ValueError
            except ValueError:
                review_exclusions["malformed_review"] += 1
                continue
            valid_review_rows += 1
            candidate = _Review(row["order_id"].strip(), row["review_id"].strip(), score, creation, answer)
            existing = reviews.get(candidate.order_id)
            if existing is None or (candidate.answer, candidate.creation, candidate.review_id) > (existing.answer, existing.creation, existing.review_id):
                reviews[candidate.order_id] = candidate
    review_duplicates = valid_review_rows - len(reviews)
    review_exclusions["superseded_multiple_review"] += review_duplicates

    delays = [order.delay_days for order in orders.values()]
    late_delays = [value for value in delays if value > 0]
    early_days = [max(-value, 0.0) for value in delays if value <= 0]
    bands: dict[str, list[_Order]] = {name: [] for name in ("on_time_or_early", "slightly_late", "moderately_late", "severely_late")}
    for order in orders.values():
        bands[_severity(order.delay_days)].append(order)

    def scores_for(group: list[_Order]) -> list[int]:
        return [reviews[order.order_id].score for order in group if order.order_id in reviews]

    def band_stats(group: list[_Order]) -> DelayBandStatistics:
        scores = scores_for(group)
        return DelayBandStatistics(
            count=len(group),
            average_review_score=sum(scores) / len(scores) if scores else None,
            low_review_rate=sum(score <= 2 for score in scores) / len(scores) if scores else None,
        )

    state_groups = {
        "on_time_or_early": bands["on_time_or_early"],
        "late": bands["slightly_late"] + bands["moderately_late"] + bands["severely_late"],
    }
    review_state: dict[str, ReviewGroupStatistics] = {}
    for name, group in state_groups.items():
        scores = scores_for(group)
        counts = Counter(scores)
        review_state[name] = ReviewGroupStatistics(
            count=len(scores),
            average_score=sum(scores) / len(scores) if scores else None,
            low_review_rate=sum(score <= 2 for score in scores) / len(scores) if scores else None,
            score_counts={str(score): counts[score] for score in range(1, 6)},
        )

    profile = DeliveryCalibrationProfile(
        source=SourceMetadata(
            dataset_name="Brazilian E-Commerce Public Dataset by Olist",
            institution="Olist / Kaggle",
            source_url=OLIST_SOURCE_URL,
            period="2016 through 2018",
            usage_scope="Delivered-order lateness and order-level review association only.",
            license_or_use_statement="Official Kaggle page reports CC BY-NC-SA 4.0.",
        ),
        order_quality=_quality(order_raw, len(orders), order_exclusions, order_missing, order_duplicates, "one eligible delivered order"),
        review_quality=_quality(review_raw, len(reviews), review_exclusions, review_missing, review_duplicates, "one latest valid review per order"),
        eligible_delivered_orders=len(orders),
        late_orders=len(late_delays),
        late_delivery_rate=len(late_delays) / len(orders) if orders else 0.0,
        delivery_delay_days=summarize(delays),
        late_delay_days=summarize(late_delays),
        early_or_on_time_days=summarize(early_days),
        severity_bands={name: band_stats(group) for name, group in bands.items()},
        reviews_by_delivery_state=review_state,
        reviewed_eligible_orders=sum(len(scores_for(group)) for group in state_groups.values()),
    )
    lateness_summary = tuple(
        {"band": name, **stats.model_dump()} for name, stats in profile.severity_bands.items()
    )
    review_summary = tuple(
        {"delivery_state": name, **stats.model_dump()} for name, stats in review_state.items()
    )
    return OlistBuildResult(profile, lateness_summary, review_summary)
