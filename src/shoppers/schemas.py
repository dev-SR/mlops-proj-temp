"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ShopperFeatures(BaseModel):
    """Feature vector for a single browsing session (no Revenue target)."""

    model_config = ConfigDict(extra="forbid")

    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: str
    OperatingSystems: int | str
    Browser: int | str
    Region: int | str
    TrafficType: int | str
    VisitorType: str
    Weekend: bool | str


class PredictRequest(BaseModel):
    instances: list[ShopperFeatures] = Field(min_length=1)


class PredictInstanceResult(BaseModel):
    prediction: int
    probability: float


class PredictResponse(BaseModel):
    predictions: list[PredictInstanceResult]
    model_uri: str
    run_id: str | None = None


class TrainResponse(BaseModel):
    status: str
    run_id: str
    metrics: dict[str, Any]
    model_path: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
