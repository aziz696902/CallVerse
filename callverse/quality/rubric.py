"""Delivery-specific V1 quality rubric and deterministic score arithmetic."""

from __future__ import annotations

from collections.abc import Mapping

from .models import QualityDimension, QualityDimensionScore, QualityWeights

RUBRIC_VERSION = "callverse-delivery-v1"

DEFAULT_WEIGHTS = QualityWeights(
    factual_accuracy=0.25,
    procedure_adherence=0.20,
    compliance=0.20,
    relevance=0.15,
    customer_satisfaction=0.10,
    sentiment_handling=0.10,
)

RUBRIC: dict[QualityDimension, dict[int, str]] = {
    QualityDimension.FACTUAL_ACCURACY: {
        1: "Invents or contradicts customer-specific order, tracking, payment, or refund facts.",
        2: "Makes important claims with weak evidence or materially misstates available evidence.",
        3: "Core facts are supported, with minor omissions or imprecision that does not mislead.",
        4: "All material customer facts are explicitly grounded in business-tool evidence.",
        5: "Grounding is complete, precise, transparent, and distinguishes facts from estimates.",
    },
    QualityDimension.RELEVANCE: {
        1: "Does not address the delivery/e-commerce request or answers a different question.",
        2: "Touches the topic but misses the customer's primary need.",
        3: "Answers the main request adequately, though some content is generic or incomplete.",
        4: "Directly addresses the request with useful, concise next steps.",
        5: "Fully tailored, efficient, and anticipates the next relevant customer question.",
    },
    QualityDimension.PROCEDURE_ADHERENCE: {
        1: "Bypasses a required safety or operational procedure with serious consequences.",
        2: "Misses a required verification, approval, policy, or escalation step.",
        3: "Follows the essential delivery-support procedure with minor gaps.",
        4: "Correctly follows verification, tool, policy, and escalation steps.",
        5: "Follows all required steps and documents a clear, auditable path forward.",
    },
    QualityDimension.COMPLIANCE: {
        1: "Performs an unauthorized refund/account action or another critical violation.",
        2: "Creates meaningful operational risk through a missing approval or unsafe action.",
        3: "No material violation, but compliance evidence or safeguards are incomplete.",
        4: "Required approvals, privacy boundaries, and authorization limits are respected.",
        5: "Compliance is exemplary, explicit, and fully supported by observable evidence.",
    },
    QualityDimension.CUSTOMER_SATISFACTION: {
        1: "Likely worsens the experience or leaves the customer without a viable path.",
        2: "Provides little useful help or leaves substantial uncertainty.",
        3: "Provides an acceptable answer or correct escalation path.",
        4: "Likely leaves the customer informed, supported, and clear on next steps.",
        5: "Delivers an exceptional recovery or resolution without making unsupported promises.",
    },
    QualityDimension.SENTIMENT_HANDLING: {
        1: "Dismisses, mocks, or aggravates frustration or anger.",
        2: "Responds coldly or ignores clear emotional cues.",
        3: "Maintains a neutral professional tone without strong emotional adaptation.",
        4: "Acknowledges the customer's emotion and responds calmly and appropriately.",
        5: "De-escalates skillfully with specific empathy while preserving factual boundaries.",
    },
}


def calculate_overall_score(
    scores: Mapping[QualityDimension, QualityDimensionScore],
    weights: QualityWeights = DEFAULT_WEIGHTS,
) -> float:
    missing = set(QualityDimension) - set(scores)
    if missing:
        raise ValueError(f"missing quality dimensions: {sorted(item.value for item in missing)}")
    total = sum(scores[dimension].score * weight for dimension, weight in weights.as_dict().items())
    return round(total, 2)


def render_rubric() -> str:
    lines = [
        f"Rubric version: {RUBRIC_VERSION}",
        (
            "Scores: 1 serious failure; 2 significant problem; 3 acceptable baseline; "
            "4 strong; 5 exceptional."
        ),
        "Weights are expert-defined V1 values, not learned from data.",
    ]
    weights = DEFAULT_WEIGHTS.as_dict()
    for dimension in QualityDimension:
        lines.append(f"\n{dimension.value} (weight {weights[dimension]:.2f})")
        for score, definition in RUBRIC[dimension].items():
            lines.append(f"{score}: {definition}")
    return "\n".join(lines)
