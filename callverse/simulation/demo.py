"""Quick offline comparison: ``python -m callverse.simulation.demo``."""

from __future__ import annotations

from callverse.scenarios import get_scenario

from .engine import compare_scenarios


def _percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.1%}"


def _minutes(value: float | None) -> str:
    return "n/a" if value is None else f"{value:8.2f}"


def main() -> None:
    names = ("normal_day", "rainy_peak", "staff_shortage", "perfect_storm")
    rows = compare_scenarios(tuple(get_scenario(name) for name in names), seed=42)
    print("Scenario         Requests  Avg Wait      SLA  Abandon  Occupancy")
    for row in rows:
        print(
            f"{row.scenario_name:<17}"
            f"{row.generated_requests:>8}  "
            f"{_minutes(row.average_waiting_time):>8}  "
            f"{_percentage(row.sla):>7}  "
            f"{_percentage(row.abandonment_rate):>7}  "
            f"{_percentage(row.occupancy):>9}"
        )


if __name__ == "__main__":
    main()
