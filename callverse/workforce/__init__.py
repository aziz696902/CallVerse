"""CallVerse Workforce Manager and Erlang-C analytical baseline."""

from .erlang_c import evaluate_erlang_c, minimum_agents, offered_load, utilization
from .manager import (
    build_workforce_plan,
    compare_staffing_strategies,
    default_workforce_config,
    summarize_plan,
)
from .models import WorkforceConfig, WorkforcePlan

__all__ = [
    "WorkforceConfig",
    "WorkforcePlan",
    "build_workforce_plan",
    "compare_staffing_strategies",
    "default_workforce_config",
    "evaluate_erlang_c",
    "minimum_agents",
    "offered_load",
    "summarize_plan",
    "utilization",
]
