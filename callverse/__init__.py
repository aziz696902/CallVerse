"""Stable domain contracts for CallVerse components."""

from .advisor import Advisor, HelpPilotAdvisor, map_helppilot_result
from .domain import (
    AdvisorResult,
    Channel,
    CustomerPersona,
    CustomerProfile,
    CustomerTier,
    KnowledgeBaseState,
    KpiSnapshot,
    RequestIntent,
    ScenarioConfig,
    SupportRequest,
    Urgency,
)
from .scenarios import SCENARIO_PRESETS, get_scenario

__all__ = [
    "Advisor",
    "AdvisorResult",
    "Channel",
    "CustomerPersona",
    "CustomerProfile",
    "CustomerTier",
    "HelpPilotAdvisor",
    "KnowledgeBaseState",
    "KpiSnapshot",
    "RequestIntent",
    "SCENARIO_PRESETS",
    "ScenarioConfig",
    "SupportRequest",
    "Urgency",
    "get_scenario",
    "map_helppilot_result",
]
