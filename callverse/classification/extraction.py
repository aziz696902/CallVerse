"""Deterministic enrichment independent of statistical intent classification."""

from __future__ import annotations

import re

from callverse.domain import Urgency


_ORDER_PATTERNS = (
    re.compile(r"\bORD[-\s]?([A-Z0-9]{3,})\b", re.IGNORECASE),
    re.compile(r"\border(?:\s+(?:number|no\.?|id))?\s*[:#-]?\s*([0-9]{3,})\b", re.IGNORECASE),
    re.compile(r"(?<!\w)#([0-9]{3,})\b"),
)


def extract_order_id(text: str) -> str | None:
    for pattern in _ORDER_PATTERNS:
        match = pattern.search(text)
        if match:
            return f"ORD-{match.group(1).upper()}"
    return None


def infer_urgency(text: str) -> Urgency:
    """Transparent provisional rules; Bitext provides no urgency ground truth."""

    normalized = text.casefold()
    if re.search(r"\b(emergency|danger|unsafe|safety issue)\b", normalized):
        return Urgency.CRITICAL
    high_signals = (
        r"\b(urgent|urgently|asap|immediately)\b",
        r"\b(cancel|cancellation)\b.*\b(today|deadline|before it ships|right now)\b",
        r"\b(lost|never arrived|never showed up|missing package)\b",
        r"\b(already complained|complained twice|again and again|repeatedly)\b",
    )
    return Urgency.HIGH if any(re.search(pattern, normalized) for pattern in high_signals) else Urgency.NORMAL
