"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.shoppers.schemas import PredictRequest, ShopperFeatures, TrainResponse


def test_shopper_features_ok() -> None:
    feat = ShopperFeatures(
        Administrative=1,
        Administrative_Duration=1.0,
        Informational=0,
        Informational_Duration=0.0,
        ProductRelated=2,
        ProductRelated_Duration=3.0,
        BounceRates=0.1,
        ExitRates=0.2,
        PageValues=0.0,
        SpecialDay=0.0,
        Month="Nov",
        OperatingSystems=1,
        Browser=1,
        Region=1,
        TrafficType=1,
        VisitorType="New_Visitor",
        Weekend=False,
    )
    assert feat.Month == "Nov"


def test_predict_request_requires_instance() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(instances=[])


def test_train_response() -> None:
    resp = TrainResponse(status="ok", run_id="abc", metrics={"f1": 0.5}, model_path="/m")
    assert resp.run_id == "abc"
