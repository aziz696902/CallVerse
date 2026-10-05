from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from callverse.domain import CustomerPersona, RequestIntent, ScenarioConfig
from callverse.scenarios import SCENARIO_PRESETS, get_scenario
from callverse.simulation import DEFAULT_POLICY, RequestOutcome, run_simulation
from callverse.simulation.policies import SimulationPolicy, TriangularMinutes


def scenario_variant(name: str, **updates: object) -> ScenarioConfig:
    values = get_scenario("normal_day").model_dump()
    values.update({"name": name, **updates})
    return ScenarioConfig.model_validate(values)


class SimulationExecutionTests(unittest.TestCase):
    def test_same_scenario_and_seed_are_reproducible(self) -> None:
        scenario = get_scenario("normal_day")
        first = run_simulation(scenario, seed=1234)
        second = run_simulation(scenario, seed=1234)
        self.assertEqual(first, second)

    def test_all_seven_presets_run(self) -> None:
        for scenario in SCENARIO_PRESETS:
            with self.subTest(scenario=scenario.name):
                result = run_simulation(scenario)
                self.assertEqual(result.scenario_name, scenario.name)
                self.assertGreater(result.counts.generated, 0)

    def test_kpis_and_waits_have_possible_values(self) -> None:
        result = run_simulation(get_scenario("perfect_storm"), seed=12)
        for record in result.request_records:
            self.assertGreaterEqual(record.waiting_time, 0)
        self.assertGreaterEqual(result.kpis.abandonment_rate, 0)
        self.assertLessEqual(result.kpis.abandonment_rate, 1)
        self.assertIsNotNone(result.kpis.occupancy)
        self.assertGreaterEqual(result.kpis.occupancy, 0)
        self.assertLessEqual(result.kpis.occupancy, 1)
        self.assertLessEqual(result.busy_advisor_minutes, result.capacity_advisor_minutes)
        self.assertIsNotNone(result.kpis.sla)
        self.assertGreaterEqual(result.kpis.sla, 0)
        self.assertLessEqual(result.kpis.sla, 1)

    def test_request_accounting_is_consistent(self) -> None:
        for scenario in SCENARIO_PRESETS:
            result = run_simulation(scenario, seed=77)
            counts = result.counts
            self.assertEqual(
                counts.generated,
                counts.completed + counts.abandoned + counts.remaining,
            )
            self.assertEqual(sum(result.intent_counts.values()), counts.generated)
            self.assertEqual(sum(result.persona_counts.values()), counts.generated)

    def test_simulation_does_not_call_live_help_pilot_advisor(self) -> None:
        with patch(
            "callverse.advisor.HelpPilotAdvisor.handle_support_request",
            side_effect=AssertionError("live advisor must not be called"),
        ):
            result = run_simulation(get_scenario("normal_day"), seed=991)
        self.assertGreater(result.counts.generated, 0)

    def test_snapshots_are_compact_and_include_horizon(self) -> None:
        result = run_simulation(get_scenario("normal_day"), seed=22)
        self.assertEqual(result.snapshots[0].simulation_time, 0)
        self.assertEqual(result.snapshots[-1].simulation_time, result.duration)
        self.assertLessEqual(len(result.snapshots), result.duration / 15 + 1)


class SimulationBehaviorTests(unittest.TestCase):
    def test_arrival_profile_is_normalized_over_scenario_horizon(self) -> None:
        policy = SimulationPolicy(arrival_slot_multipliers=tuple(range(1, 49)))
        duration = 487.0
        start = 473
        mean = policy.mean_arrival_multiplier(duration, start)
        elapsed = 0.0
        weighted = 0.0
        while elapsed < duration:
            minute = (start + elapsed) % 1440
            segment = min(30 - minute % 30, duration - elapsed)
            weighted += policy.normalized_arrival_multiplier(elapsed, start, duration) * segment
            elapsed += segment
        self.assertAlmostEqual(mean, weighted * mean / duration)
        self.assertAlmostEqual(weighted / duration, 1.0)

    def test_severe_staff_reduction_does_not_improve_queue_performance(self) -> None:
        well_staffed = scenario_variant(
            "well_staffed", simulation_duration=240, demand_multiplier=1.4, available_agents=9
        )
        short_staffed = scenario_variant(
            "short_staffed", simulation_duration=240, demand_multiplier=1.4, available_agents=2
        )
        good = run_simulation(well_staffed, seed=987)
        short = run_simulation(short_staffed, seed=987)
        self.assertEqual(good.counts.generated, short.counts.generated)
        self.assertGreaterEqual(short.counts.abandoned, good.counts.abandoned)
        self.assertLessEqual(short.kpis.sla, good.kpis.sla)
        self.assertGreaterEqual(short.kpis.average_waiting_time, good.kpis.average_waiting_time)

    def test_high_demand_creates_greater_operational_pressure(self) -> None:
        low = scenario_variant(
            "low_demand", simulation_duration=240, demand_multiplier=0.7, available_agents=5
        )
        high = scenario_variant(
            "high_demand", simulation_duration=240, demand_multiplier=2.2, available_agents=5
        )
        low_result = run_simulation(low, seed=654)
        high_result = run_simulation(high, seed=654)
        self.assertGreater(high_result.counts.generated, low_result.counts.generated)
        self.assertGreaterEqual(high_result.counts.abandoned, low_result.counts.abandoned)
        self.assertGreaterEqual(high_result.kpis.occupancy, low_result.kpis.occupancy)

    def test_intent_and_persona_mix_influence_generated_traffic(self) -> None:
        scenario = scenario_variant(
            "distribution_check",
            simulation_duration=2_000,
            available_agents=50,
            customer_persona_mix={
                CustomerPersona.LOYAL: 0.65,
                CustomerPersona.AT_RISK: 0.35,
            },
            request_intent_mix={
                RequestIntent.TRACKING: 0.70,
                RequestIntent.GENERAL: 0.30,
            },
        )
        result = run_simulation(scenario, seed=2468)
        total = result.counts.generated
        self.assertAlmostEqual(result.intent_counts[RequestIntent.TRACKING] / total, 0.70, delta=0.05)
        self.assertAlmostEqual(result.persona_counts[CustomerPersona.LOYAL] / total, 0.65, delta=0.05)

    def test_overload_with_low_patience_produces_clean_abandonments(self) -> None:
        tiny_patience = {
            persona: TriangularMinutes(0.01, 0.02, 0.03) for persona in CustomerPersona
        }
        policy = replace(DEFAULT_POLICY, patience_times=tiny_patience)
        overloaded = scenario_variant(
            "overloaded", simulation_duration=60, demand_multiplier=5.0, available_agents=1
        )
        result = run_simulation(overloaded, seed=1357, policy=policy)
        self.assertGreater(result.counts.abandoned, 0)
        abandoned = [
            record for record in result.request_records if record.outcome is RequestOutcome.ABANDONED
        ]
        self.assertTrue(abandoned)
        self.assertTrue(all(record.service_start is None for record in abandoned))

    def test_request_types_have_distinct_handling_profiles(self) -> None:
        tracking = DEFAULT_POLICY.handling_times[RequestIntent.TRACKING]
        complaint = DEFAULT_POLICY.handling_times[RequestIntent.COMPLAINT]
        self.assertLess(tracking.mode, complaint.mode)
        self.assertLess(tracking.maximum, complaint.maximum)

    def test_personas_have_transparent_patience_differences(self) -> None:
        loyal = DEFAULT_POLICY.patience_times[CustomerPersona.LOYAL]
        at_risk = DEFAULT_POLICY.patience_times[CustomerPersona.AT_RISK]
        self.assertGreater(loyal.mode, at_risk.mode)

    def test_result_leaves_unmodeled_kpis_empty(self) -> None:
        kpis = run_simulation(get_scenario("normal_day"), seed=44).kpis
        self.assertIsNone(kpis.first_contact_resolution)
        self.assertIsNone(kpis.customer_satisfaction)
        self.assertIsNone(kpis.operating_cost)

    def test_policy_rejects_missing_intent_or_persona_profiles(self) -> None:
        with self.assertRaises(ValueError):
            SimulationPolicy(handling_times={})
        with self.assertRaises(ValueError):
            SimulationPolicy(patience_times={})


if __name__ == "__main__":
    unittest.main()
