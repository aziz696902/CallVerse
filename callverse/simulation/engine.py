"""SimPy support-center engine driven entirely by ``ScenarioConfig``."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from random import Random

import simpy

from callverse.domain import (
    Channel,
    CustomerPersona,
    CustomerProfile,
    KpiSnapshot,
    RequestIntent,
    ScenarioConfig,
    SupportRequest,
)

from .models import (
    RequestCounts,
    RequestEventRecord,
    RequestOutcome,
    ScenarioComparison,
    SimulationResult,
    TimeSeriesSnapshot,
)
from .policies import DEFAULT_POLICY, SimulationPolicy


_MESSAGES: dict[RequestIntent, str] = {
    RequestIntent.TRACKING: "Where is my order?",
    RequestIntent.REFUND: "I would like a refund.",
    RequestIntent.DAMAGED_ITEM: "My item arrived damaged.",
    RequestIntent.ADDRESS_CHANGE: "Please change my delivery address.",
    RequestIntent.CANCEL_ORDER: "Please cancel my order.",
    RequestIntent.PAYMENT_ISSUE: "I have a payment problem.",
    RequestIntent.COMPLAINT: "I need to make a complaint.",
    RequestIntent.GENERAL: "I need help with a general question.",
}

_ORDER_INTENTS = {
    RequestIntent.TRACKING,
    RequestIntent.REFUND,
    RequestIntent.DAMAGED_ITEM,
    RequestIntent.ADDRESS_CHANGE,
    RequestIntent.CANCEL_ORDER,
}


@dataclass
class _RequestState:
    request: SupportRequest
    customer: CustomerProfile
    patience_minutes: float
    handling_minutes: float
    status: str = "queued"
    service_start: float | None = None
    end_time: float | None = None
    waiting_time: float | None = None


def _weighted_choice(rng: Random, probabilities: dict) -> object:
    values = list(probabilities)
    return rng.choices(values, weights=[probabilities[value] for value in values], k=1)[0]


class _SupportCenterSimulation:
    def __init__(
        self,
        scenario: ScenarioConfig,
        seed: int,
        policy: SimulationPolicy,
    ) -> None:
        self.scenario = scenario
        self.seed = seed
        self.policy = policy
        self.env = simpy.Environment()
        self.advisors = simpy.Resource(self.env, capacity=scenario.available_agents)
        # Separate streams keep the generated workload stable across staffing changes.
        self.arrival_rng = Random(seed)
        self.attribute_rng = Random(seed + 1)
        self.service_rng = Random(seed + 2)
        self.requests: list[_RequestState] = []
        self.snapshots: list[TimeSeriesSnapshot] = []

    def run(self) -> SimulationResult:
        self.env.process(self._generate_arrivals())
        self.env.process(self._record_snapshots())
        self.env.run(until=self.scenario.simulation_duration)
        self._append_snapshot(self.scenario.simulation_duration)
        return self._build_result()

    def _generate_arrivals(self):
        base_rate = self.policy.base_arrival_rate_per_minute * self.scenario.demand_multiplier
        if self.policy.arrival_slot_multipliers is None:
            yield from self._generate_flat_arrivals(base_rate)
            return

        profile_mean = self.policy.mean_arrival_multiplier(
            self.scenario.simulation_duration,
            self.scenario.simulation_start_minute_of_day,
        )
        maximum_multiplier = max(self.policy.arrival_slot_multipliers) / profile_mean
        candidate_rate = base_rate * maximum_multiplier
        while True:
            interarrival = self.arrival_rng.expovariate(candidate_rate)
            if self.env.now + interarrival >= self.scenario.simulation_duration:
                return
            yield self.env.timeout(interarrival)
            current_multiplier = self.policy.arrival_multiplier(
                self.env.now, self.scenario.simulation_start_minute_of_day
            ) / profile_mean
            if self.arrival_rng.random() > current_multiplier / maximum_multiplier:
                continue
            state = self._new_request(len(self.requests) + 1)
            self.requests.append(state)
            self.env.process(self._handle_request(state))

    def _generate_flat_arrivals(self, rate: float):
        while True:
            interarrival = self.arrival_rng.expovariate(rate)
            if self.env.now + interarrival >= self.scenario.simulation_duration:
                return
            yield self.env.timeout(interarrival)
            state = self._new_request(len(self.requests) + 1)
            self.requests.append(state)
            self.env.process(self._handle_request(state))

    def _new_request(self, sequence: int) -> _RequestState:
        intent = _weighted_choice(self.attribute_rng, self.scenario.request_intent_mix)
        persona = _weighted_choice(self.attribute_rng, self.scenario.customer_persona_mix)
        assert isinstance(intent, RequestIntent)
        assert isinstance(persona, CustomerPersona)

        patience = self.policy.patience_times[persona].sample(self.attribute_rng)
        handling = self.policy.handling_times[intent].sample(self.service_rng)
        request_id = f"{self.scenario.name}-REQ-{sequence:06d}"
        customer_id = f"{self.scenario.name}-CUST-{sequence:06d}"
        order_id = f"ORD-{sequence:06d}" if intent in _ORDER_INTENTS else None

        request = SupportRequest(
            request_id=request_id,
            customer_id=customer_id,
            order_id=order_id,
            intent=intent,
            channel=Channel.TEXT,
            customer_message=_MESSAGES[intent],
            arrival_time=self.env.now,
        )
        customer = CustomerProfile(
            customer_id=customer_id,
            persona=persona,
            patience_seconds=patience * 60,
        )
        return _RequestState(request, customer, patience, handling)

    def _handle_request(self, state: _RequestState):
        with self.advisors.request() as advisor_request:
            patience_timeout = self.env.timeout(state.patience_minutes)
            outcome = yield advisor_request | patience_timeout

            if advisor_request not in outcome:
                state.status = "abandoned"
                state.waiting_time = self.env.now - state.request.arrival_time
                state.end_time = self.env.now
                return

            state.status = "in_service"
            state.service_start = self.env.now
            state.waiting_time = self.env.now - state.request.arrival_time
            yield self.env.timeout(state.handling_minutes)
            state.status = "completed"
            state.end_time = self.env.now

    def _record_snapshots(self):
        while True:
            self._append_snapshot(self.env.now)
            yield self.env.timeout(self.policy.snapshot_interval_minutes)

    def _append_snapshot(self, simulation_time: float) -> None:
        snapshot = TimeSeriesSnapshot(
            simulation_time=simulation_time,
            queue_size=len(self.advisors.queue),
            busy_agents=self.advisors.count,
            completed_count=sum(state.status == "completed" for state in self.requests),
            abandoned_count=sum(state.status == "abandoned" for state in self.requests),
        )
        if self.snapshots and self.snapshots[-1].simulation_time == simulation_time:
            self.snapshots[-1] = snapshot
        else:
            self.snapshots.append(snapshot)

    def _build_result(self) -> SimulationResult:
        generated = len(self.requests)
        completed_states = [state for state in self.requests if state.status == "completed"]
        abandoned_states = [state for state in self.requests if state.status == "abandoned"]
        served_states = [state for state in self.requests if state.service_start is not None]
        observed_wait_states = [
            state for state in self.requests if state.waiting_time is not None
        ]

        completed = len(completed_states)
        abandoned = len(abandoned_states)
        remaining = generated - completed - abandoned
        busy_minutes = sum(
            max(
                0.0,
                min(
                    self.scenario.simulation_duration,
                    state.service_start + state.handling_minutes,
                )
                - state.service_start,
            )
            for state in served_states
            if state.service_start is not None
        )
        capacity_minutes = self.scenario.available_agents * self.scenario.simulation_duration

        average_wait = (
            sum(state.waiting_time or 0.0 for state in observed_wait_states)
            / len(observed_wait_states)
            if observed_wait_states
            else None
        )
        average_handling = (
            sum(state.handling_minutes for state in completed_states) / completed
            if completed
            else None
        )
        sla = (
            sum((state.waiting_time or 0.0) <= self.policy.sla_target_minutes for state in served_states)
            / len(served_states)
            if served_states
            else None
        )

        kpis = KpiSnapshot(
            queue_size=len(self.advisors.queue),
            average_waiting_time=average_wait,
            sla=sla,
            abandonment_rate=abandoned / generated if generated else 0.0,
            average_handling_time=average_handling,
            first_contact_resolution=None,
            occupancy=busy_minutes / capacity_minutes,
            customer_satisfaction=None,
            operating_cost=None,
        )

        intent_counts = Counter(state.request.intent for state in self.requests)
        persona_counts = Counter(state.customer.persona for state in self.requests)
        record_limit = self.policy.max_event_records
        request_records = tuple(
            self._event_record(state) for state in self.requests[:record_limit]
        )

        return SimulationResult(
            scenario_name=self.scenario.name,
            seed=self.seed,
            duration=self.scenario.simulation_duration,
            available_agents=self.scenario.available_agents,
            ai_advisor_enabled=self.scenario.ai_advisor_enabled,
            sla_target_minutes=self.policy.sla_target_minutes,
            busy_advisor_minutes=busy_minutes,
            capacity_advisor_minutes=capacity_minutes,
            counts=RequestCounts(
                generated=generated,
                completed=completed,
                abandoned=abandoned,
                remaining=remaining,
            ),
            kpis=kpis,
            intent_counts={intent: intent_counts[intent] for intent in RequestIntent},
            persona_counts={persona: persona_counts[persona] for persona in CustomerPersona},
            snapshots=tuple(self.snapshots),
            request_records=request_records,
            event_records_truncated=generated > record_limit,
        )

    def _event_record(self, state: _RequestState) -> RequestEventRecord:
        if state.status == "completed":
            outcome = RequestOutcome.COMPLETED
        elif state.status == "abandoned":
            outcome = RequestOutcome.ABANDONED
        else:
            outcome = RequestOutcome.REMAINING

        waiting_time = state.waiting_time
        if waiting_time is None:
            waiting_time = self.scenario.simulation_duration - state.request.arrival_time

        return RequestEventRecord(
            request_id=state.request.request_id,
            customer_id=state.request.customer_id,
            persona=state.customer.persona,
            intent=state.request.intent,
            arrival_time=state.request.arrival_time,
            patience_minutes=state.patience_minutes,
            handling_minutes=state.handling_minutes,
            service_start=state.service_start,
            end_time=state.end_time,
            waiting_time=waiting_time,
            outcome=outcome,
        )


def run_simulation(
    scenario: ScenarioConfig,
    *,
    seed: int | None = None,
    policy: SimulationPolicy = DEFAULT_POLICY,
) -> SimulationResult:
    """Run one scenario entirely in memory and return a validated result."""

    actual_seed = scenario.random_seed if seed is None else seed
    if actual_seed < 0:
        raise ValueError("seed must be non-negative")
    return _SupportCenterSimulation(scenario, actual_seed, policy).run()


def compare_scenarios(
    scenarios: list[ScenarioConfig] | tuple[ScenarioConfig, ...],
    *,
    seed: int | None = None,
    policy: SimulationPolicy = DEFAULT_POLICY,
) -> tuple[ScenarioComparison, ...]:
    """Run several scenarios and return compact KPI rows for comparison."""

    rows = []
    for scenario in scenarios:
        result = run_simulation(scenario, seed=seed, policy=policy)
        rows.append(
            ScenarioComparison(
                scenario_name=result.scenario_name,
                generated_requests=result.counts.generated,
                average_waiting_time=result.kpis.average_waiting_time,
                sla=result.kpis.sla,
                abandonment_rate=result.kpis.abandonment_rate or 0.0,
                occupancy=result.kpis.occupancy or 0.0,
            )
        )
    return tuple(rows)
