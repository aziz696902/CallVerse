# CallVerse Domain Contracts

## Purpose

The `callverse` package defines the small, validated vocabulary shared by future
CallVerse components. It prevents the simulator, advisor, analytics, and planning
layers from exchanging loosely structured dictionaries or depending directly on
HelpPilot internals.

The package does not implement simulation, customer generation, analytics, or
optimization.

## Main Contracts

- `CustomerProfile` identifies a simulated customer by persona, optional tier, and
  optional patience.
- `SupportRequest` carries the request and simulated arrival time presented to an
  advisor.
- `AdvisorResult` reports resolution, escalation, automation, response, and handling
  duration without embedding future quality scores.
- `KpiSnapshot` defines the promised center-level metrics. Every metric defaults to
  `None` until a future KPI engine computes real values.
- `ScenarioConfig` describes duration, demand, staffing, customer and intent mixes,
  external conditions, delivery disruption, knowledge state, and AI availability.
  It rejects invalid rates, staffing, durations, and probability mixes.

## Advisor Boundary

`Advisor` is a Python protocol exposing:

```python
handle_support_request(request, customer) -> AdvisorResult
```

The future simulator can depend on this protocol without knowing about LangGraph,
Streamlit, Chroma, or Groq. `HelpPilotAdvisor` is a thin adapter around HelpPilot's
existing public `run_turn` function. Its result mapping and injected-runner path are
tested offline; invoking the default live HelpPilot runner still requires the existing
Groq configuration and may pause for human refund approval.

## Intended Flow

```text
ScenarioConfig
      ↓
Client Simulator (future)
      ↓
SupportRequest
      ↓
Advisor Interface
      ↓
AdvisorResult
      ↓
Digital Twin / KPI Engine (future)
```

## Generic Scenarios

Scenarios are not hardcoded around rain, load, or agent count. The same contract also
varies customer behavior, request intent, delivery reliability, external conditions,
knowledge-base health, and AI availability. Seven presets demonstrate this range:
`normal_day`, `rainy_peak`, `flash_sale`, `staff_shortage`, `customer_crisis`,
`knowledge_failure`, and `perfect_storm`.

All preset values are illustrative prototype parameters. They have not been derived
from operational research or real contact-center datasets and must be calibrated in a
later data-focused phase before being interpreted as realistic.
