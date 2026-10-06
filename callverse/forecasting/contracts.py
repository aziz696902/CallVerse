"""Typed contracts for CallVerse demand forecasting."""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise

from pydantic import Field, field_validator, model_validator

from callverse.domain import DomainModel


class ForecastPoint(DomainModel):
    timestamp: datetime
    predicted_contacts: float = Field(ge=0)


class DemandForecast(DomainModel):
    forecast_origin: datetime
    horizon: int = Field(gt=0)
    interval_minutes: int = Field(gt=0)
    points: tuple[ForecastPoint, ...]
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)

    @field_validator("points")
    @classmethod
    def validate_progression(cls, points: tuple[ForecastPoint, ...]):
        for previous, current in pairwise(points):
            if (current.timestamp - previous.timestamp).total_seconds() != 1800:
                raise ValueError("forecast points must progress in 30-minute intervals")
        return points

    @model_validator(mode="after")
    def validate_horizon(self):
        if len(self.points) != self.horizon:
            raise ValueError("forecast point count must match the declared horizon")
        if self.interval_minutes != 30:
            raise ValueError("CallVerse V1 forecasts must use 30-minute intervals")
        return self


class ForecastMetrics(DomainModel):
    mae: float = Field(ge=0)
    rmse: float = Field(ge=0)
    smape: float = Field(ge=0)
    mean_poisson_deviance: float | None = Field(default=None, ge=0)


class HorizonMetrics(DomainModel):
    early_0_to_2h: ForecastMetrics
    middle_2_to_12h: ForecastMetrics
    late_12_to_24h: ForecastMetrics


class ForecastEvaluation(DomainModel):
    model_name: str
    origins: int = Field(gt=0)
    observations: int = Field(gt=0)
    overall: ForecastMetrics
    by_horizon: HorizonMetrics
