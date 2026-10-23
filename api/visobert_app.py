"""
==============================================================================
ScamShield-VN: ViSoBERT Direct Inspection API
==============================================================================
Provides endpoints for direct testing & telemetry of the Tier 1 ViSoBERT model.
Uses pipeline.visobert_classifier as the core inference engine.
==============================================================================
"""

import logging
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from pipeline.visobert_classifier import get_visobert_classifier
from pipeline.persistence import save_prediction

router = APIRouter(prefix="/api/v1/visobert", tags=["ViSoBERT"])
logger = logging.getLogger("ScamShield.ViSoBERT")


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    core_check_id: Optional[int] = Field(None, gt=0, description="Core check_requests.id used as the idempotency key")

    @field_validator("message")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace text")
        return value


class PredictResponse(BaseModel):
    model_label: Literal["HAM", "SCAM"]
    decision: Literal["HAM", "SCAM", "REVIEW"]
    scam_probability: float = Field(..., ge=0, le=1)
    confidence: float = Field(..., ge=0, le=1)
    temperature: float
    threshold_low: float
    threshold_high: float
    latency_ms: float
    engine: str = "onnx"


def load_model():
    """Validates and loads the ViSoBERT model via pipeline singleton."""
    return get_visobert_classifier()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    request_id = uuid.uuid4().hex[:8]
    logger.info("ViSoBERT prediction requested: id=%s chars=%d", request_id, len(request.message))
    try:
        classifier = get_visobert_classifier()
        result = classifier.predict(request.message)
        try:
            save_prediction(message=request.message, core_check_id=request.core_check_id, result=result)
        except Exception as exc:
            logger.exception("ViSoBERT result persistence failed: id=%s", request_id)
            raise HTTPException(
                status_code=503,
                detail="Inference completed but the result could not be saved to the database.",
            ) from exc
        logger.info(
            "ViSoBERT prediction complete: id=%s engine=%s prob=%.4f decision=%s lat_ms=%.2f",
            request_id, result["engine"], result["scam_probability"], result["decision"], result["latency_ms"]
        )
        return PredictResponse(
            model_label=result["model_label"],
            decision=result["decision"],
            scam_probability=result["scam_probability"],
            confidence=result["confidence"],
            temperature=result["temperature"],
            threshold_low=result["threshold_low"],
            threshold_high=result["threshold_high"],
            latency_ms=result["latency_ms"],
            engine=result["engine"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ViSoBERT prediction error: id=%s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
