"""
==============================================================================
ScamShield-VN: ViSoBERT Classifier (Tier 1 Production Engine)
==============================================================================
Provides high-performance text classification for Vietnamese SMS and chat.
Supports:
  1. ONNX Runtime CPU Fast-Path (< 35ms latency, ultra-lightweight RAM)
  2. PyTorch SafeTensors (Local/CUDA fallback)
  3. Safe Semantic / Heuristic Rule-Based Fallback (Zero-crash resilience)
==============================================================================
"""

import os
import json
import time
import math
import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("ScamShield.ViSoBERTClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

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
        self.engine_type = "heuristic_fallback"
        self.temperature = 1.4015
        self.threshold_low = 0.35
        self.threshold_high = 0.85
        self.max_length = 128
        self.device_str = "cpu"

        # Model / Session handles
        self.ort_session = None
        self.torch_model = None
        self.tokenizer = None

        # Compile heuristic fallback regex
        self._init_heuristics()

        # Load configuration & weights
        self._load_config_and_weights()

    def _init_heuristics(self):
        self.scam_patterns = [
            r"(?i)(trúng\s*thưởng|trung\s*thuong|nhận\s*quà|nhan\s*qua|tri\s*ân|tri\s*an)",
            r"(?i)(khóa\s*tài\s*khoản|khoa\s*tai\s*khoan|tạm\s*khóa|tam\s*khoa|xác\s*thực|xac\s*thuc|cập\s*nhật\s*sinh\s*trắc)",
            r"(?i)(bộ\s*công\s*an|bo\s*cong\s*an|viện\s*kiểm\s*sát|vien\s*kiem\s*sat|lệnh\s*bắt|lenh\s*bat|tạm\s*giam|tam\s*giam|rửa\s*tiền|rua\s*tien)",
            r"(?i)(việc\s*nhẹ\s*lương\s*cao|viec\s*nhe|tuyển\s*ctv|tuyen\s*ctv|đánh\s*giá\s*sản\s*phẩm|hoa\s*hồng)",
            r"(?i)(vay\s*tiền|vay\s*tien|duyệt\s*ngay|duyet\s*ngay|không\s*cần\s*thế\s*chấp|khong\s*can\s*the\s*chap|lãi\s*suất\s*0%)",
            r"(?i)(link\s*đăng\s*nhập|link\s*dang\s*nhap|truy\s*cập\s*link|truy\s*cap\s*link|nhập\s*mã\s*otp|nhap\s*ma\s*otp|nhắn\s*zalo)",
            r"(?i)(https?://[^\s]+\.(xyz|top|vip|cc|tk|ml|ga|cf|gq|info|online|site))",
        ]
        self.compiled_scam_regex = [re.compile(p) for p in self.scam_patterns]

        self.ham_patterns = [
            r"(?i)(mã\s*otp|ma\s*otp|otp\s*của\s*quý\s*khách|otp\s*xac\s*thuc|hieu\s*luc\s*trong\s*\d+\s*phut|không\s*chia\s*sẻ|khong\s*chia\s*se)",
            r"(?i)(tài\s*khoản\s*thanh\s*toán|tiền\s*điện|tiền\s*nước|tien\s*dien|tien\s*nuoc|vcb\s*digibank|digibank|shopeepay|vietcombank)",
            r"(?i)(gói\s*cước|goi\s*cuoc|gia\s*hạn\s*thành\s*công|data\s*sử\s*dụng|quy\s*khach|thue\s*bao)",
            r"(?i)(chuyen\s*khoan\s*nhanh|giao\s*dich\s*thanh\s*cong|bien\s*dong\s*so\s*du|so\s*du\s*tk)",
            r"(?i)(cảm\s*ơn\s*quý\s*khách|cam\s*on\s*quy\s*khach|ăn\s*lẩu|đi\s*chơi|cà\s*phê|alo\s*nhé)",
        ]
        self.compiled_ham_regex = [re.compile(p) for p in self.ham_patterns]

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
                logger.warning("PyTorch initialization failed (%s). Falling back to heuristic engine.", e)

        # 4. Fallback: Semantic Rule Engine (Engine C)
        logger.warning(
            "ViSoBERT binary model weights not found or are un-smudged Git LFS pointers in %s. "
            "Operating with high-fidelity semantic heuristic engine. "
            "To activate full 110M neural model, run 'git lfs pull'.",
            self.model_dir
        )
        self.engine_type = "heuristic_fallback"

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

    def _predict_heuristic(self, message: str) -> Tuple[float, List[str]]:
        scam_score = 0.0
        ham_score = 0.0
        matched_cues = []

        for p in self.compiled_scam_regex:
            m = p.search(message)
            if m:
                scam_score += 2.2
                matched_cues.append(f"ScamPattern[{m.group(0)[:30]}]")

        for p in self.compiled_ham_regex:
            m = p.search(message)
            if m:
                ham_score += 1.8
                matched_cues.append(f"HamPattern[{m.group(0)[:30]}]")

        urls = re.findall(r"https?://[^\s]+", message)
        for u in urls:
            if any(tld in u for tld in [".xyz", ".top", ".vip", ".cc", "bit.ly"]):
                scam_score += 2.5
                matched_cues.append(f"SuspiciousDomain[{u}]")

        diff = scam_score - ham_score
        calibrated_logit = diff / self.temperature
        scam_prob = 1.0 / (1.0 + math.exp(-calibrated_logit))
        return scam_prob, matched_cues

    def predict(self, message: str) -> Dict[str, Any]:
        """
        Classifies a message and returns calibrated probabilities and tri-decision verdict.
        """
        started = time.perf_counter()
        matched_cues = []

        if self.engine_type == "onnx" and self.ort_session is not None:
            scam_prob = self._predict_onnx(message)
        elif self.engine_type == "pytorch" and self.torch_model is not None:
            scam_prob = self._predict_pytorch(message)
        else:
            scam_prob, matched_cues = self._predict_heuristic(message)

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
            "matched_cues": matched_cues,
        }


@lru_cache(maxsize=1)
def get_visobert_classifier() -> ViSoBERTClassifier:
    """Singleton getter for the ViSoBERT classifier."""
    return ViSoBERTClassifier()
