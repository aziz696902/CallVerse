"""Pure playback state and bounded visual models for comparative replay."""

from __future__ import annotations

from pydantic import Field, field_validator

from callverse.domain import DomainModel

from .comparative_replay import ComparativeFrame, ComparativeReplay

SUPPORTED_SPEEDS = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_SPEED = 4.0
ONE_X_SECONDS_PER_FRAME = 2.0


class PlaybackState(DomainModel):
    frame_index: int = Field(default=0, ge=0)
    is_playing: bool = False
    speed_multiplier: float = DEFAULT_SPEED
    reached_end: bool = False

    @field_validator("speed_multiplier")
    @classmethod
    def validate_speed(cls, value: float) -> float:
        if value not in SUPPORTED_SPEEDS:
            supported = ", ".join(f"{speed:g}x" for speed in SUPPORTED_SPEEDS)
            raise ValueError(f"unsupported replay speed; choose {supported}")
        return value


class CappedMarkers(DomainModel):
    markers: tuple[str, ...]
    hidden_count: int = Field(ge=0)
    total_count: int = Field(ge=0)


def _validate_timeline(total_frames: int) -> None:
    if total_frames <= 0:
        raise ValueError("playback requires at least one frame")


def initial_playback_state(total_frames: int) -> PlaybackState:
    _validate_timeline(total_frames)
    return PlaybackState()


def play(state: PlaybackState, total_frames: int) -> PlaybackState:
    _validate_timeline(total_frames)
    if state.frame_index >= total_frames:
        raise IndexError("playback frame index is outside the timeline")
    if state.is_playing or state.frame_index == total_frames - 1:
        return state
    return state.model_copy(update={"is_playing": True, "reached_end": False})


def pause(state: PlaybackState) -> PlaybackState:
    if not state.is_playing:
        return state
    return state.model_copy(update={"is_playing": False})


def restart(state: PlaybackState, total_frames: int) -> PlaybackState:
    _validate_timeline(total_frames)
    return state.model_copy(
        update={"frame_index": 0, "is_playing": False, "reached_end": False}
    )


def advance(state: PlaybackState, total_frames: int) -> PlaybackState:
    _validate_timeline(total_frames)
    if state.frame_index >= total_frames:
        raise IndexError("playback frame index is outside the timeline")
    if not state.is_playing:
        return state
    next_index = min(state.frame_index + 1, total_frames - 1)
    reached_end = next_index == total_frames - 1
    return state.model_copy(
        update={
            "frame_index": next_index,
            "is_playing": not reached_end,
            "reached_end": reached_end,
        }
    )


def scrub(
    state: PlaybackState, frame_index: int, total_frames: int
) -> PlaybackState:
    _validate_timeline(total_frames)
    if not 0 <= frame_index < total_frames:
        raise IndexError("manual replay position is outside the timeline")
    return state.model_copy(
        update={
            "frame_index": frame_index,
            "is_playing": False,
            "reached_end": frame_index == total_frames - 1,
        }
    )


def set_speed(state: PlaybackState, speed_multiplier: float) -> PlaybackState:
    if speed_multiplier not in SUPPORTED_SPEEDS:
        supported = ", ".join(f"{speed:g}x" for speed in SUPPORTED_SPEEDS)
        raise ValueError(f"unsupported replay speed; choose {supported}")
    if state.speed_multiplier == speed_multiplier:
        return state
    return state.model_copy(update={"speed_multiplier": speed_multiplier})


def stop_for_stale_comparison(state: PlaybackState) -> PlaybackState:
    return pause(state)


def seconds_per_frame(speed_multiplier: float) -> float:
    if speed_multiplier not in SUPPORTED_SPEEDS:
        supported = ", ".join(f"{speed:g}x" for speed in SUPPORTED_SPEEDS)
        raise ValueError(f"unsupported replay speed; choose {supported}")
    return ONE_X_SECONDS_PER_FRAME / speed_multiplier


def current_frame(
    replay: ComparativeReplay, state: PlaybackState
) -> ComparativeFrame:
    if not 0 <= state.frame_index < len(replay.frames):
        raise IndexError("playback frame index is outside the comparison timeline")
    return replay.frames[state.frame_index]


def queue_markers(queue_size: int, *, cap: int = 10) -> CappedMarkers:
    if queue_size < 0:
        raise ValueError("queue size cannot be negative")
    if cap <= 0:
        raise ValueError("queue marker cap must be positive")
    visible = min(queue_size, cap)
    return CappedMarkers(
        markers=("WAITING",) * visible,
        hidden_count=queue_size - visible,
        total_count=queue_size,
    )


def advisor_markers(
    busy_agents: int, free_agents: int, *, cap: int = 10
) -> CappedMarkers:
    if busy_agents < 0 or free_agents < 0:
        raise ValueError("advisor counts cannot be negative")
    if cap <= 0:
        raise ValueError("advisor marker cap must be positive")
    total = busy_agents + free_agents
    visible_busy = min(busy_agents, cap)
    visible_free = min(free_agents, cap - visible_busy)
    markers = ("BUSY",) * visible_busy + ("FREE",) * visible_free
    return CappedMarkers(
        markers=markers,
        hidden_count=total - len(markers),
        total_count=total,
    )
