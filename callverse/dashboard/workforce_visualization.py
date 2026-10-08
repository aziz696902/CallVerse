"""Pure view models for the fair Dynamic Workforce replay."""

from __future__ import annotations

from pydantic import Field, model_validator

from callverse.domain import DomainModel
from callverse.forecasting.contracts import DemandForecast
from callverse.simulation import StaffingSchedule, TimeSeriesSnapshot
from callverse.workforce.dynamic import WorkforcePolicyComparison


class WorkforceVisualFrame(DomainModel):
    index: int = Field(ge=0)
    frame_count: int = Field(gt=0)
    simulation_minute: float = Field(ge=0)
    simulated_clock: str = Field(pattern=r"^\d{2}:\d{2}$")
    forecast_contacts: float = Field(ge=0)
    baseline_advisors: int = Field(ge=1)
    callverse_advisors: int = Field(ge=1)
    baseline_agent_hours: float = Field(ge=0)
    callverse_agent_hours: float = Field(ge=0)
    baseline: TimeSeriesSnapshot
    callverse: TimeSeriesSnapshot
    baseline_staffing_event: str | None = None
    staffing_event: str | None = None
    resource_explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_synchronized_frame(self) -> WorkforceVisualFrame:
        if self.index >= self.frame_count:
            raise ValueError("workforce frame index is outside the timeline")
        if self.baseline.simulation_time != self.callverse.simulation_time:
            raise ValueError("workforce snapshots are not synchronized")
        if self.simulation_minute != self.baseline.simulation_time:
            raise ValueError("workforce frame time does not match its snapshots")
        return self


def simulated_clock(minute: float) -> str:
    """Format elapsed simulation time, preserving the 24:00 final boundary."""

    rounded = round(minute)
    hours, minutes = divmod(rounded, 60)
    return f"{hours:02d}:{minutes:02d}"


def cumulative_agent_hours(schedule: StaffingSchedule, minute: float) -> float:
    """Integrate scheduled capacity from minute zero through ``minute``."""

    bounded = min(max(minute, 0.0), schedule.horizon_minutes)
    return sum(
        slot.advisors
        * max(0.0, min(bounded, slot.end_minute) - slot.start_minute)
        / 60
        for slot in schedule.slots
        if slot.start_minute < bounded
    )


def _slot_index(schedule: StaffingSchedule, minute: float) -> int:
    if minute >= schedule.horizon_minutes:
        return len(schedule.slots) - 1
    return min(int(minute // schedule.slot_minutes), len(schedule.slots) - 1)


def _resource_explanation(
    baseline_advisors: int,
    callverse_advisors: int,
    forecast_contacts: float,
    *,
    staffing_event: str | None,
) -> str:
    if callverse_advisors > baseline_advisors:
        reason = (
            "CallVerse is using more of the shared daily staffing budget at this moment "
            f"as forecast demand reaches {forecast_contacts:.2f} contacts."
        )
    elif callverse_advisors < baseline_advisors:
        reason = (
            "The uniform baseline is using more of the shared daily staffing budget at "
            f"this moment while forecast demand is {forecast_contacts:.2f} contacts."
        )
    else:
        reason = (
            f"Both policies expose {baseline_advisors} advisors while forecast demand is "
            f"{forecast_contacts:.2f} contacts."
        )
    reason += " Both strategies have the same total daily staffing budget."
    if staffing_event is not None:
        reason += " The scheduled boundary changes capacity without interrupting active contacts."
    return reason


def build_workforce_visual_frames(
    comparison: WorkforcePolicyComparison,
    forecast: DemandForecast,
) -> tuple[WorkforceVisualFrame, ...]:
    """Join stored synchronized snapshots to the predefined staffing plan."""

    baseline_snapshots = comparison.baseline_result.snapshots
    callverse_snapshots = comparison.callverse_result.snapshots
    if len(baseline_snapshots) != len(callverse_snapshots):
        raise ValueError("workforce comparison snapshot counts differ")
    event_by_minute = {
        event.simulation_minute: event.title for event in comparison.staffing_events
    }
    baseline_event_by_minute = {
        event.simulation_minute: event.title
        for event in comparison.baseline_staffing_events
    }
    frames = []
    frame_count = len(baseline_snapshots)
    for index, (baseline, callverse) in enumerate(
        zip(baseline_snapshots, callverse_snapshots, strict=True)
    ):
        minute = baseline.simulation_time
        if minute != callverse.simulation_time:
            raise ValueError("workforce comparison snapshots are not synchronized")
        slot_index = _slot_index(comparison.callverse_schedule, minute)
        forecast_contacts = forecast.points[slot_index].predicted_contacts
        baseline_advisors = comparison.baseline_schedule.slots[slot_index].advisors
        callverse_advisors = comparison.callverse_schedule.slots[slot_index].advisors
        staffing_event = event_by_minute.get(minute)
        frames.append(
            WorkforceVisualFrame(
                index=index,
                frame_count=frame_count,
                simulation_minute=minute,
                simulated_clock=simulated_clock(minute),
                forecast_contacts=forecast_contacts,
                baseline_advisors=baseline_advisors,
                callverse_advisors=callverse_advisors,
                baseline_agent_hours=cumulative_agent_hours(
                    comparison.baseline_schedule, minute
                ),
                callverse_agent_hours=cumulative_agent_hours(
                    comparison.callverse_schedule, minute
                ),
                baseline=baseline,
                callverse=callverse,
                baseline_staffing_event=baseline_event_by_minute.get(minute),
                staffing_event=staffing_event,
                resource_explanation=_resource_explanation(
                    baseline_advisors,
                    callverse_advisors,
                    forecast_contacts,
                    staffing_event=staffing_event,
                ),
            )
        )
    return tuple(frames)


def demand_chart_rows(
    forecast: DemandForecast, current_minute: float
) -> tuple[dict[str, float], ...]:
    """Return the known 48-slot forecast with one explicit current-position series."""

    current_slot = min(int(current_minute // 30), len(forecast.points) - 1)
    return tuple(
        {
            "simulation_minute": float(index * 30),
            "Forecast contacts": point.predicted_contacts,
            "Current time": (
                point.predicted_contacts if index == current_slot else float("nan")
            ),
        }
        for index, point in enumerate(forecast.points)
    )


def staffing_step_rows(
    comparison: WorkforcePolicyComparison, current_minute: float
) -> tuple[dict[str, float], ...]:
    """Duplicate interval boundaries so Streamlit draws an exact staffing step."""

    rows: list[dict[str, float]] = []
    slots = comparison.callverse_schedule.slots
    for index, slot in enumerate(slots):
        row = {
            "simulation_minute": slot.start_minute,
            "Uniform baseline": float(
                comparison.baseline_schedule.slots[index].advisors
            ),
            "CallVerse forecast-informed": float(slot.advisors),
        }
        rows.append(row)
        rows.append(
            {
                "simulation_minute": slot.end_minute,
                "Uniform baseline": float(
                    comparison.baseline_schedule.slots[index].advisors
                ),
                "CallVerse forecast-informed": float(slot.advisors),
            }
        )
    return tuple(rows)


def queue_trajectory_rows(
    frames: tuple[WorkforceVisualFrame, ...], through_index: int
) -> tuple[dict[str, float | int], ...]:
    """Expose actual stored queue snapshots only through the inspected frame."""

    if not 0 <= through_index < len(frames):
        raise IndexError("workforce queue position is outside the timeline")
    return tuple(
        {
            "simulation_minute": frame.simulation_minute,
            "Uniform baseline": frame.baseline.queue_size,
            "CallVerse forecast-informed": frame.callverse.queue_size,
        }
        for frame in frames[: through_index + 1]
    )
