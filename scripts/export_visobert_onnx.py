"""Export the saved ScamShield ViSoBERT classifier and verify ONNX parity."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "visobert_finetuned")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    output = (args.output or model_dir / "visobert_classifier.onnx").resolve()
    if not (model_dir / "model.safetensors").is_file():
        parser.error(f"No fine-tuned model.safetensors found in {model_dir}")
    if not (model_dir / "training_report.json").is_file():
        parser.error(f"No training_report.json found in {model_dir}")

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    max_length = int(report["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True
    ).cpu().eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapper = LogitsOnly(model).eval()
    sample = tokenizer("Xac nhan giao dich ngan hang", return_tensors="pt", max_length=max_length, truncation=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (sample["input_ids"], sample["attention_mask"]),
            str(output),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                          "attention_mask": {0: "batch", 1: "sequence"},
                          "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    examples = [
        "Ma OTP cua ban la 123456. Khong chia se cho nguoi khac.",
        "Tai khoan bi khoa. Truy cap lien ket va nhap OTP ngay.",
        "Chao ban, hom nay di ca phe khong?",
    ]
    encoded = tokenizer(examples, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.inference_mode():
        reference = wrapper(encoded["input_ids"], encoded["attention_mask"]).numpy()
    inputs = {name: encoded[name].numpy() for name in ("input_ids", "attention_mask")}
    actual = session.run(["logits"], inputs)[0]
    max_abs_diff = float(np.max(np.abs(reference - actual)))
    if max_abs_diff > 1e-3:
        raise RuntimeError(f"ONNX/PyTorch logits differ by {max_abs_diff:.6g}; expected <= 0.001")
    summary = {
        "onnx_path": str(output),
        "size_bytes": output.stat().st_size,
        "opset": 17,
        "providers_tested": session.get_providers(),
        "examples_checked": len(examples),
        "max_absolute_logit_difference": max_abs_diff,
        "temperature": report["temperature"],
        "threshold_low": report["threshold_low"],
        "threshold_high": report["threshold_high"],
    }
    (output.parent / "onnx_export_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
