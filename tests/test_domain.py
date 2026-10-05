from __future__ import annotations

import unittest

from pydantic import ValidationError

from callverse import (
    Advisor,
    AdvisorResult,
    Channel,
    CustomerPersona,
    CustomerProfile,
    KpiSnapshot,
    RequestIntent,
    SCENARIO_PRESETS,
    ScenarioConfig,
    SupportRequest,
    Urgency,
    get_scenario,
)
from callverse.advisor import HelpPilotAdvisor, map_helppilot_result


VALID_PERSONA_MIX = {
    CustomerPersona.NEW: 0.2,
    CustomerPersona.LOYAL: 0.3,
    CustomerPersona.UNHAPPY: 0.2,
    CustomerPersona.PREMIUM: 0.2,
    CustomerPersona.AT_RISK: 0.1,
}
VALID_INTENT_MIX = {RequestIntent.TRACKING: 0.6, RequestIntent.GENERAL: 0.4}


def valid_scenario(**overrides: object) -> ScenarioConfig:
    values: dict[str, object] = {
        "name": "test_scenario",
        "simulation_duration": 60,
        "available_agents": 2,
        "customer_persona_mix": VALID_PERSONA_MIX,
        "request_intent_mix": VALID_INTENT_MIX,
    }
    values.update(overrides)
    return ScenarioConfig(**values)


class ScenarioConfigTests(unittest.TestCase):
    def test_valid_scenario_is_accepted(self) -> None:
        scenario = valid_scenario()
        self.assertEqual(scenario.available_agents, 2)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            valid_scenario(late_delivery_rate=1.01)

    def test_zero_or_negative_agent_count_is_rejected(self) -> None:
        for count in (0, -1):
            with self.subTest(count=count), self.assertRaises(ValidationError):
                valid_scenario(available_agents=count)

    def test_duration_and_demand_must_be_positive(self) -> None:
        for field in ("simulation_duration", "demand_multiplier"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                valid_scenario(**{field: 0})

    def test_customer_mix_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sum to 1.0"):
            valid_scenario(customer_persona_mix={CustomerPersona.NEW: 0.8})

    def test_request_mix_must_sum_to_one(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sum to 1.0"):
            valid_scenario(request_intent_mix={RequestIntent.TRACKING: 0.9})

    def test_each_mix_probability_must_be_bounded(self) -> None:
        with self.assertRaisesRegex(ValidationError, "between 0 and 1"):
            valid_scenario(
                request_intent_mix={RequestIntent.TRACKING: 1.1, RequestIntent.GENERAL: -0.1}
            )

    def test_all_presets_validate_and_names_are_unique(self) -> None:
        self.assertEqual(len(SCENARIO_PRESETS), 7)
        names = [scenario.name for scenario in SCENARIO_PRESETS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(names),
            {
                "normal_day",
                "rainy_peak",
                "flash_sale",
                "staff_shortage",
                "customer_crisis",
                "knowledge_failure",
                "perfect_storm",
            },
        )
        for scenario in SCENARIO_PRESETS:
            self.assertIsInstance(ScenarioConfig.model_validate(scenario.model_dump()), ScenarioConfig)
            self.assertIs(get_scenario(scenario.name), scenario)


class DomainContractTests(unittest.TestCase):
    def test_support_request_creation(self) -> None:
        request = SupportRequest(
            request_id="REQ-1",
            customer_id="CUST-1",
            order_id="ORD-1",
            intent="tracking",
            urgency="high",
            channel="text",
            customer_message="Where is my order?",
            arrival_time=12.5,
        )
        self.assertEqual(request.intent, RequestIntent.TRACKING)
        self.assertEqual(request.channel, Channel.TEXT)

    def test_advisor_result_creation(self) -> None:
        result = AdvisorResult(
            request_id="REQ-1",
            resolved=True,
            escalated=False,
            automated=True,
            response_text="Your order is in transit.",
            handling_duration=2.5,
        )
        self.assertTrue(result.resolved)

    def test_enum_values_are_predictable(self) -> None:
        self.assertEqual(CustomerPersona.AT_RISK.value, "at_risk")
        self.assertIs(RequestIntent("refund"), RequestIntent.REFUND)
        self.assertEqual(str(Urgency.CRITICAL), "critical")

    def test_kpi_snapshot_has_no_fabricated_defaults(self) -> None:
        snapshot = KpiSnapshot()
        self.assertTrue(all(value is None for value in snapshot.model_dump().values()))


class AdvisorBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = SupportRequest(
            request_id="REQ-2",
            customer_id="CUST-2",
            intent=RequestIntent.GENERAL,
            customer_message="Hello",
            arrival_time=0,
        )

    def test_deterministic_help_pilot_mapping(self) -> None:
        mapped = map_helppilot_result(
            "REQ-2",
            {"status": "done", "state": {"final_reply": "Hello!", "escalated": False}},
            handling_duration=0.25,
        )
        self.assertEqual(mapped.response_text, "Hello!")
        self.assertTrue(mapped.resolved)
        self.assertTrue(mapped.automated)

    def test_adapter_accepts_an_offline_runner(self) -> None:
        def runner(message: str, customer_id: str, request_id: str) -> dict:
            return {"status": "done", "state": {"final_reply": f"Received: {message}"}}

        advisor = HelpPilotAdvisor(runner=runner)
        self.assertIsInstance(advisor, Advisor)
        result = advisor.handle_support_request(
            self.request,
            CustomerProfile(customer_id="CUST-2", persona=CustomerPersona.NEW),
        )
        self.assertEqual(result.response_text, "Received: Hello")

    def test_adapter_rejects_mismatched_customer_context(self) -> None:
        advisor = HelpPilotAdvisor(runner=lambda *_: {})
        with self.assertRaisesRegex(ValueError, "must match"):
            advisor.handle_support_request(
                self.request,
                CustomerProfile(customer_id="OTHER", persona=CustomerPersona.NEW),
            )


if __name__ == "__main__":
    unittest.main()
