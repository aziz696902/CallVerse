"""Dashboard-ready manager summaries over quality results."""

from __future__ import annotations

from collections import Counter
from statistics import fmean

from .models import QualityDimension, QualityEvaluationResult


def aggregate_quality(results: list[QualityEvaluationResult]) -> dict[str, object]:
    completed = [result for result in results if result.overall_score is not None]
    dimension_averages: dict[str, float | None] = {}
    for dimension in QualityDimension:
        values = [
            getattr(result, dimension.value).score
            for result in completed
            if getattr(result, dimension.value) is not None
        ]
        dimension_averages[dimension.value] = round(fmean(values), 2) if values else None

    severities = Counter(flag.severity.value for result in results for flag in result.flags)
    sentiments = Counter(
        result.customer_sentiment.value for result in results if result.customer_sentiment is not None
    )
    return {
        "interaction_count": len(results),
        "evaluated_count": len(completed),
        "unavailable_count": len(results) - len(completed),
        "average_overall_quality": (
            round(fmean(result.overall_score for result in completed), 2) if completed else None
        ),
        "average_by_dimension": dimension_averages,
        "requires_supervisor_review_count": sum(
            result.requires_supervisor_review for result in results
        ),
        "compliance_flags_by_severity": dict(sorted(severities.items())),
        "unresolved_count": sum(not result.resolved for result in results),
        "escalated_count": sum(result.escalated for result in results),
        "sentiment_distribution": dict(sorted(sentiments.items())),
    }

