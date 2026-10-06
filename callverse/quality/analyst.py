"""Quality orchestration with optional structured Groq judgment."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .guardrails import GuardrailAssessment, evaluate_guardrails
from .models import (
    QualityDimension,
    QualityDimensionScore,
    QualityEvaluationInput,
    QualityEvaluationResult,
    QualityFlag,
    QualityJudgeOutput,
    QualitySeverity,
    QualityStatus,
    QualityWeights,
)
from .rubric import (
    DEFAULT_WEIGHTS,
    RUBRIC_VERSION,
    calculate_overall_score,
    render_rubric,
)

StructuredJudge = Callable[[QualityEvaluationInput, str], QualityJudgeOutput | dict[str, Any]]


def _score_map(output: QualityJudgeOutput) -> dict[QualityDimension, QualityDimensionScore]:
    return {
        QualityDimension.FACTUAL_ACCURACY: output.factual_accuracy,
        QualityDimension.RELEVANCE: output.relevance,
        QualityDimension.PROCEDURE_ADHERENCE: output.procedure_adherence,
        QualityDimension.COMPLIANCE: output.compliance,
        QualityDimension.CUSTOMER_SATISFACTION: output.customer_satisfaction,
        QualityDimension.SENTIMENT_HANDLING: output.sentiment_handling,
    }


def _apply_caps(
    scores: dict[QualityDimension, QualityDimensionScore], assessment: GuardrailAssessment
) -> dict[QualityDimension, QualityDimensionScore]:
    adjusted = dict(scores)
    for dimension, cap in assessment.score_caps.items():
        current = adjusted[dimension]
        if current.score > cap:
            adjusted[dimension] = QualityDimensionScore(
                score=cap,
                justification=(
                    f"{current.justification} Deterministic guardrail capped this dimension at {cap}."
                ),
            )
    return adjusted


def _merge_flags(hard: tuple[QualityFlag, ...], judged: tuple[QualityFlag, ...]) -> tuple[QualityFlag, ...]:
    merged: list[QualityFlag] = list(hard)
    keys = {(flag.code, flag.evidence_reference) for flag in hard}
    for flag in judged:
        key = (flag.code, flag.evidence_reference)
        if key not in keys:
            merged.append(flag)
            keys.add(key)
    return tuple(merged)


class GroqStructuredQualityJudge:
    """Use the same ChatGroq provider/configuration family as HelpPilot."""

    def __init__(self, model: str | None = None, timeout: int = 60) -> None:
        self.model = model
        self.timeout = timeout

    def __call__(self, evidence: QualityEvaluationInput, rubric: str) -> QualityJudgeOutput:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        from helppilot import config

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is unavailable")
        llm = ChatGroq(
            model=self.model or config.REVIEWER_MODEL,
            temperature=0,
            timeout=self.timeout,
            api_key=config.GROQ_API_KEY,
        ).with_structured_output(QualityJudgeOutput)
        system = (
            "You are the CallVerse delivery-support quality judge. Score only from the "
            "observable evidence and rubric. Never invent tool facts. Appropriate escalation "
            "is a valid procedural outcome. Return the six scores, concise justifications, "
            "meaningful operational flags only, sentiment, and confidence. Do not calculate "
            "an overall score.\n\n" + rubric
        )
        payload = evidence.model_dump(mode="json")
        return llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=json.dumps(payload, indent=2))]
        )


class QualityAnalyst:
    def __init__(
        self,
        judge: StructuredJudge | None = None,
        *,
        evaluator_type: str = "injected_structured",
        evaluator_provider: str | None = None,
        weights: QualityWeights = DEFAULT_WEIGHTS,
    ) -> None:
        self._judge = judge
        self._evaluator_type = evaluator_type if judge is not None else "none"
        self._evaluator_provider = evaluator_provider
        self._weights = weights

    @classmethod
    def with_groq(cls, model: str | None = None) -> QualityAnalyst:
        return cls(
            GroqStructuredQualityJudge(model=model),
            evaluator_type="groq_structured",
            evaluator_provider="groq",
        )

    def _unavailable(
        self,
        evidence: QualityEvaluationInput,
        assessment: GuardrailAssessment,
        reason: str,
    ) -> QualityEvaluationResult:
        return QualityEvaluationResult(
            interaction_id=evidence.interaction_id,
            status=QualityStatus.UNAVAILABLE,
            flags=assessment.flags,
            requires_supervisor_review=True,
            evaluator_type="none" if self._judge is None else self._evaluator_type,  # type: ignore[arg-type]
            evaluator_provider=self._evaluator_provider,
            rubric_version=RUBRIC_VERSION,
            guardrail_observations=assessment.observations,
            status_reason=reason,
            resolved=evidence.resolved,
            escalated=evidence.escalated,
        )

    def evaluate(self, evidence: QualityEvaluationInput) -> QualityEvaluationResult:
        assessment = evaluate_guardrails(evidence)
        if self._judge is None:
            return self._unavailable(evidence, assessment, "structured quality judge not configured")
        try:
            raw = self._judge(evidence, render_rubric())
            judged = QualityJudgeOutput.model_validate(raw)
        except Exception as exc:  # noqa: BLE001 - external judge failures must fail safely
            return self._unavailable(
                evidence,
                assessment,
                f"structured quality judge unavailable: {type(exc).__name__}",
            )

        scores = _apply_caps(_score_map(judged), assessment)
        flags = _merge_flags(assessment.flags, judged.flags)
        severe = any(flag.severity in {QualitySeverity.HIGH, QualitySeverity.CRITICAL} for flag in flags)
        requires_review = severe or any(score.score <= 2 for score in scores.values())
        return QualityEvaluationResult(
            interaction_id=evidence.interaction_id,
            status=QualityStatus.COMPLETED,
            factual_accuracy=scores[QualityDimension.FACTUAL_ACCURACY],
            relevance=scores[QualityDimension.RELEVANCE],
            procedure_adherence=scores[QualityDimension.PROCEDURE_ADHERENCE],
            compliance=scores[QualityDimension.COMPLIANCE],
            customer_satisfaction=scores[QualityDimension.CUSTOMER_SATISFACTION],
            sentiment_handling=scores[QualityDimension.SENTIMENT_HANDLING],
            overall_score=calculate_overall_score(scores, self._weights),
            flags=flags,
            requires_supervisor_review=requires_review,
            evaluator_type=self._evaluator_type,  # type: ignore[arg-type]
            evaluator_provider=self._evaluator_provider,
            rubric_version=RUBRIC_VERSION,
            customer_sentiment=judged.customer_sentiment,
            evaluator_confidence=judged.confidence,
            guardrail_observations=assessment.observations,
            resolved=evidence.resolved,
            escalated=evidence.escalated,
        )

