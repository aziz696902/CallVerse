"""Illustrative scenario presets; these values are not dataset-calibrated yet."""

from __future__ import annotations

from .domain import CustomerPersona, KnowledgeBaseState, RequestIntent, ScenarioConfig


def _personas(**values: float) -> dict[CustomerPersona, float]:
    return {CustomerPersona(key): value for key, value in values.items()}


def _intents(**values: float) -> dict[RequestIntent, float]:
    return {RequestIntent(key): value for key, value in values.items()}


BALANCED_PERSONAS = _personas(new=0.20, loyal=0.30, unhappy=0.15, premium=0.20, at_risk=0.15)
BALANCED_INTENTS = _intents(
    tracking=0.30,
    refund=0.15,
    damaged_item=0.10,
    address_change=0.10,
    cancel_order=0.10,
    payment_issue=0.08,
    complaint=0.07,
    general=0.10,
)


SCENARIO_PRESETS: tuple[ScenarioConfig, ...] = (
    ScenarioConfig(
        name="normal_day",
        description="Baseline workload, delivery conditions, and customer mix.",
        simulation_duration=480,
        random_seed=101,
        external_condition="normal",
        demand_multiplier=1.0,
        late_delivery_rate=0.08,
        available_agents=8,
        customer_persona_mix=BALANCED_PERSONAS,
        request_intent_mix=BALANCED_INTENTS,
    ),
    ScenarioConfig(
        name="rainy_peak",
        description="Heavy rain raises late deliveries and tracking contacts.",
        simulation_duration=480,
        random_seed=202,
        external_condition="heavy_rain",
        demand_multiplier=1.35,
        late_delivery_rate=0.30,
        available_agents=8,
        customer_persona_mix=_personas(
            new=0.15, loyal=0.25, unhappy=0.25, premium=0.15, at_risk=0.20
        ),
        request_intent_mix=_intents(
            tracking=0.50,
            refund=0.15,
            damaged_item=0.08,
            address_change=0.05,
            cancel_order=0.05,
            payment_issue=0.04,
            complaint=0.08,
            general=0.05,
        ),
    ),
    ScenarioConfig(
        name="flash_sale",
        description="A promotion creates high contact volume across many intents.",
        simulation_duration=360,
        random_seed=303,
        external_condition="flash_sale",
        demand_multiplier=2.20,
        late_delivery_rate=0.12,
        available_agents=10,
        customer_persona_mix=_personas(
            new=0.40, loyal=0.20, unhappy=0.10, premium=0.15, at_risk=0.15
        ),
        request_intent_mix=_intents(
            tracking=0.20,
            refund=0.12,
            damaged_item=0.08,
            address_change=0.12,
            cancel_order=0.16,
            payment_issue=0.15,
            complaint=0.07,
            general=0.10,
        ),
    ),
    ScenarioConfig(
        name="staff_shortage",
        description="Moderately elevated demand with substantially fewer advisors.",
        simulation_duration=480,
        random_seed=404,
        external_condition="staff_absence",
        demand_multiplier=1.15,
        late_delivery_rate=0.10,
        available_agents=3,
        customer_persona_mix=BALANCED_PERSONAS,
        request_intent_mix=BALANCED_INTENTS,
    ),
    ScenarioConfig(
        name="customer_crisis",
        description="Customer sentiment deteriorates, increasing complaints and refunds.",
        simulation_duration=480,
        random_seed=505,
        external_condition="service_incident",
        demand_multiplier=1.30,
        late_delivery_rate=0.25,
        available_agents=7,
        customer_persona_mix=_personas(
            new=0.05, loyal=0.10, unhappy=0.40, premium=0.15, at_risk=0.30
        ),
        request_intent_mix=_intents(
            tracking=0.15,
            refund=0.25,
            damaged_item=0.08,
            address_change=0.04,
            cancel_order=0.05,
            payment_issue=0.04,
            complaint=0.35,
            general=0.04,
        ),
    ),
    ScenarioConfig(
        name="knowledge_failure",
        description="Normal operations with an outdated advisor knowledge base.",
        simulation_duration=480,
        random_seed=606,
        external_condition="normal",
        demand_multiplier=1.0,
        late_delivery_rate=0.08,
        available_agents=8,
        customer_persona_mix=BALANCED_PERSONAS,
        request_intent_mix=BALANCED_INTENTS,
        knowledge_base_state=KnowledgeBaseState.OUTDATED,
    ),
    ScenarioConfig(
        name="perfect_storm",
        description="Severe demand, delivery disruption, difficult customers, and low staffing.",
        simulation_duration=480,
        random_seed=707,
        external_condition="severe_delivery_disruption",
        demand_multiplier=2.50,
        late_delivery_rate=0.50,
        available_agents=4,
        customer_persona_mix=_personas(
            new=0.05, loyal=0.10, unhappy=0.40, premium=0.10, at_risk=0.35
        ),
        request_intent_mix=_intents(
            tracking=0.32,
            refund=0.25,
            damaged_item=0.08,
            address_change=0.03,
            cancel_order=0.06,
            payment_issue=0.04,
            complaint=0.18,
            general=0.04,
        ),
        knowledge_base_state=KnowledgeBaseState.PARTIAL,
    ),
)


_SCENARIOS_BY_NAME = {scenario.name: scenario for scenario in SCENARIO_PRESETS}


def get_scenario(name: str) -> ScenarioConfig:
    """Return a named preset, failing clearly when the name is unknown."""

    try:
        return _SCENARIOS_BY_NAME[name]
    except KeyError as exc:
        available = ", ".join(sorted(_SCENARIOS_BY_NAME))
        raise KeyError(f"unknown scenario {name!r}; available: {available}") from exc
