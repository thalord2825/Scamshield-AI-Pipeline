"""Routes for testing the fine-tuned ViSoBERT classifier."""

import json
import logging
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("VISOBERT_MODEL_DIR", ROOT / "models" / "visobert_finetuned"))
router = APIRouter(prefix="/api/v1/visobert", tags=["ViSoBERT"])
logger = logging.getLogger("ScamShield.ViSoBERT")


class PredictRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)

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


@lru_cache(maxsize=1)
def load_model():
    started = time.perf_counter()
    logger.info("Loading fine-tuned ViSoBERT checkpoint from %s", MODEL_DIR)
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report_path = MODEL_DIR / "training_report.json"
    if not report_path.is_file() or not (MODEL_DIR / "model.safetensors").is_file():
        raise FileNotFoundError(f"Fine-tuned checkpoint or training report missing in {MODEL_DIR}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    temperature = float(report["temperature"])
    low = float(report["threshold_low"])
    high = float(report["threshold_high"])
    if not temperature > 0 or not 0 < low < high < 1:
        raise ValueError("Invalid calibration or thresholds in training_report.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, use_safetensors=True
    ).eval()
    if model.config.id2label != {0: "HAM", 1: "SCAM"}:
        raise ValueError("Unexpected label mapping in fine-tuned checkpoint")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(
        "ViSoBERT ready: device=%s max_length=%d temperature=%.4f thresholds=(%.2f, %.2f) load_ms=%.2f",
        device, int(report["max_length"]), temperature, low, high,
        (time.perf_counter() - started) * 1000,
    )
    return tokenizer, model, device, temperature, low, high, int(report["max_length"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    started = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    logger.info("Prediction started: request_id=%s message_chars=%d", request_id, len(request.message))
    try:
        tokenizer, model, device, temperature, low, high, max_length = load_model()
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        logger.exception("Model unavailable: request_id=%s", request_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    import torch

    try:
        tokenization_started = time.perf_counter()
        encoded = tokenizer(request.message, return_tensors="pt", truncation=True, max_length=max_length)
        tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
        token_count = int(encoded["attention_mask"].sum().item())
        logger.info(
            "Tokenization complete: request_id=%s tokens=%d max_length=%d tokenization_ms=%.2f",
            request_id, token_count, max_length, tokenization_ms,
        )
        inference_started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**{key: value.to(device) for key, value in encoded.items()}).logits
            probability = float(torch.softmax(logits / temperature, dim=-1)[0, 1].item())
        inference_ms = (time.perf_counter() - inference_started) * 1000
    except Exception:
        logger.exception("Prediction failed: request_id=%s", request_id)
        raise
    decision = "HAM" if probability < low else "SCAM" if probability > high else "REVIEW"
    total_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "Prediction complete: request_id=%s device=%s scam_probability=%.4f decision=%s inference_ms=%.2f total_ms=%.2f",
        request_id, device, probability, decision, inference_ms, total_ms,
    )
    return PredictResponse(
        model_label="SCAM" if probability >= 0.5 else "HAM",
        decision=decision,
        scam_probability=probability,
        confidence=max(probability, 1 - probability),
        temperature=temperature,
        threshold_low=low,
        threshold_high=high,
        latency_ms=round(total_ms, 2),
    )
