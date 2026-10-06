"""Auditable mapping from Bitext source intents to CallVerse intents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from callverse.domain import RequestIntent


class MappingStrength(str, Enum):
    DIRECT = "DIRECT"
    COMBINED = "COMBINED"
    UNSUPPORTED = "UNSUPPORTED / EXCLUDED"


@dataclass(frozen=True)
class IntentMapping:
    target: RequestIntent | None
    strength: MappingStrength
    reason: str


BITEXT_INTENT_MAPPING: dict[str, IntentMapping] = {
    "cancel_order": IntentMapping(RequestIntent.CANCEL_ORDER, MappingStrength.DIRECT, "Explicit order cancellation."),
    "change_shipping_address": IntentMapping(RequestIntent.ADDRESS_CHANGE, MappingStrength.DIRECT, "Explicit shipping-address change."),
    "complaint": IntentMapping(RequestIntent.COMPLAINT, MappingStrength.DIRECT, "Explicit complaint."),
    "payment_issue": IntentMapping(RequestIntent.PAYMENT_ISSUE, MappingStrength.DIRECT, "Explicit payment problem."),
    "track_order": IntentMapping(RequestIntent.TRACKING, MappingStrength.DIRECT, "Explicit order tracking."),
    "check_refund_policy": IntentMapping(RequestIntent.REFUND, MappingStrength.COMBINED, "Refund-policy request belongs to the broader refund intent."),
    "get_refund": IntentMapping(RequestIntent.REFUND, MappingStrength.COMBINED, "Explicit refund request."),
    "track_refund": IntentMapping(RequestIntent.REFUND, MappingStrength.COMBINED, "Refund status belongs to the broader refund intent."),
}


def map_source_intent(source_intent: str) -> RequestIntent | None:
    mapping = BITEXT_INTENT_MAPPING.get(source_intent)
    return mapping.target if mapping else None
