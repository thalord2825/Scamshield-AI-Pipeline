"""
==============================================================================
ScamShield-VN: ViSoBERT Classifier (Tier 1 Production Engine)
==============================================================================
Provides high-performance text classification for Vietnamese SMS and chat using
ONNX Runtime or PyTorch model inference.
==============================================================================
"""

import os
import json
import time
import math
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger("ScamShield.ViSoBERTClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # This module owns its console handler. Stop records reaching the root
    # handler configured by api.app, which would print each inference log twice.
    logger.propagate = False

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "models" / "visobert_finetuned"


def is_valid_binary_model(file_path: Path, min_size_bytes: int = 100 * 1024) -> bool:
    """Checks whether a model file exists, is non-empty, and is not a Git LFS pointer."""
    if not file_path.is_file():
        return False
    try:
        size = file_path.stat().st_size
        if size < min_size_bytes:
            return False
        with file_path.open("rb") as f:
            header = f.read(64)
            if header.startswith(b"version https://git-lfs"):
                return False
        return True
    except OSError:
        return False


class ViSoBERTClassifier:
    """
    Tier-1 Classifier executing ViSoBERT with Temperature Calibration
    and Tri-decision Banding (HAM / REVIEW / SCAM).
    """

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = Path(model_dir or os.getenv("VISOBERT_MODEL_DIR", DEFAULT_MODEL_DIR)).resolve()
        self.engine_type = "unavailable"
        self.temperature = 1.4015
        self.threshold_low = 0.35
        self.threshold_high = 0.85
        self.max_length = 128
        self.device_str = "cpu"

        # Model / Session handles
        self.ort_session = None
        self.torch_model = None
        self.tokenizer = None

        # Load configuration & weights
        self._load_config_and_weights()

    def _load_config_and_weights(self):
        # 1. Load parameters from training_report.json or config.json
        report_path = self.model_dir / "training_report.json"
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.temperature = float(report.get("temperature", self.temperature))
                self.threshold_low = float(report.get("threshold_low", self.threshold_low))
                self.threshold_high = float(report.get("threshold_high", self.threshold_high))
                self.max_length = int(report.get("max_length", self.max_length))
                logger.info(
                    "ViSoBERT config loaded from %s: T=%.4f, thresholds=(%.2f, %.2f), max_len=%d",
                    report_path.name, self.temperature, self.threshold_low, self.threshold_high, self.max_length
                )
            except Exception as e:
                logger.warning("Error reading %s: %s", report_path, e)

        # 2. Try loading ONNX Runtime Session (Engine A)
        onnx_candidates = [
            self.model_dir / "visobert_classifier.onnx",
            self.model_dir / "visobert_tier1.onnx",
            ROOT_DIR / "models" / "visobert_tier1.onnx",
        ]
        valid_onnx = next((p for p in onnx_candidates if is_valid_binary_model(p)), None)

        if valid_onnx:
            try:
                import onnxruntime as ort
                from transformers import AutoTokenizer
                logger.info("Initializing ONNX Runtime inference session with %s", valid_onnx)
                sess_options = ort.SessionOptions()
                sess_options.intra_op_num_threads = min(4, os.cpu_count() or 1)
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.ort_session = ort.InferenceSession(str(valid_onnx), sess_options, providers=["CPUExecutionProvider"])
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
                self.engine_type = "onnx"
                self.device_str = "cpu"
                logger.info("ONNX Runtime engine successfully initialized for ViSoBERT.")
                return
            except Exception as e:
                logger.warning("ONNX initialization failed (%s). Attempting PyTorch fallback.", e)

        # 3. Try loading PyTorch SafeTensors (Engine B)
        safetensor_candidates = [
            self.model_dir / "model.safetensors",
            ROOT_DIR / "models" / "model.safetensors",
        ]
        valid_safetensor = next((p for p in safetensor_candidates if is_valid_binary_model(p)), None)

        if valid_safetensor:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                logger.info("Initializing PyTorch inference with %s", valid_safetensor)
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(self.model_dir), local_files_only=True, use_safetensors=True
                ).eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                self.torch_model = model
                self.device_str = str(device)
                self.engine_type = "pytorch"
                logger.info("PyTorch engine successfully initialized on device %s.", self.device_str)
                return
            except Exception as e:
                logger.warning("PyTorch initialization failed: %s", e)

        logger.warning(
            "ViSoBERT model weights are unavailable or are un-smudged Git LFS pointers in %s. "
            "Run 'git lfs pull' and ensure ONNX Runtime or PyTorch dependencies are installed.",
            self.model_dir
        )

    def _predict_onnx(self, message: str) -> float:
        encoded = self.tokenizer(
            message,
            return_tensors="np",
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        inputs = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        logits = self.ort_session.run(["logits"], inputs)[0]
        calibrated_logits = logits[0] / self.temperature
        exp_logits = [math.exp(z - max(calibrated_logits)) for z in calibrated_logits]
        scam_prob = exp_logits[1] / sum(exp_logits)
        return float(scam_prob)

    def _predict_pytorch(self, message: str) -> float:
        import torch
        encoded = self.tokenizer(
            message,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        with torch.inference_mode():
            device = next(self.torch_model.parameters()).device
            inputs = {k: v.to(device) for k, v in encoded.items()}
            logits = self.torch_model(**inputs).logits
            scam_prob = float(torch.softmax(logits / self.temperature, dim=-1)[0, 1].item())
        return scam_prob

    def predict(self, message: str) -> Dict[str, Any]:
        """
        Classifies a message and returns calibrated probabilities and tri-decision verdict.
        """
        started = time.perf_counter()
        logger.info(
            "ViSoBERT inference started: engine=%s device=%s message_chars=%d",
            self.engine_type, self.device_str, len(message),
        )
        if self.engine_type == "onnx" and self.ort_session is not None:
            scam_prob = self._predict_onnx(message)
        elif self.engine_type == "pytorch" and self.torch_model is not None:
            scam_prob = self._predict_pytorch(message)
        else:
            logger.error("ViSoBERT inference unavailable: no ONNX or PyTorch model is loaded")
            raise RuntimeError(
                "ViSoBERT inference is unavailable: no ONNX or PyTorch model loaded. "
                "Run 'git lfs pull' and ensure the inference dependencies are installed."
            )

        # Apply Decision Thresholds
        if scam_prob < self.threshold_low:
            decision = "HAM"
        elif scam_prob > self.threshold_high:
            decision = "SCAM"
        else:
            decision = "REVIEW"

        confidence = max(scam_prob, 1.0 - scam_prob)
        model_label = "SCAM" if scam_prob >= 0.5 else "HAM"
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "ViSoBERT inference completed: engine=%s device=%s decision=%s model_label=%s "
            "scam_probability=%.4f confidence=%.4f latency_ms=%.2f",
            self.engine_type, self.device_str, decision, model_label,
            scam_prob, confidence, latency_ms,
        )

        return {
            "decision": decision,
            "label": decision,
            "model_label": model_label,
            "class_id": 1 if model_label == "SCAM" else 0,
            "scam_probability": round(scam_prob, 4),
            "calibrated_scam_prob": round(scam_prob, 4),
            "confidence": round(confidence, 4),
            "temperature": self.temperature,
            "threshold_low": self.threshold_low,
            "threshold_high": self.threshold_high,
            "latency_ms": round(latency_ms, 2),
            "engine": self.engine_type,
            "device": self.device_str,
        }


@lru_cache(maxsize=1)
def get_visobert_classifier() -> ViSoBERTClassifier:
    """Singleton getter for the ViSoBERT classifier."""
    return ViSoBERTClassifier()
