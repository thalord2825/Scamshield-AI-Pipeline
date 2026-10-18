"""Fine-tune ViSoBERT on ScamShield's binary Vietnamese SMS data.

The supplied test CSV overlaps the training CSV. Every test message is removed
from the training pool before the validation split is made.
"""

import argparse
import csv
import hashlib
import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def key(message):
    return unicodedata.normalize("NFKC", message).strip().casefold()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"message", "label"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain message and label columns")
        rows = []
        for number, row in enumerate(reader, 2):
            message = (row.get("message") or "").strip()
            label = (row.get("label") or "").strip()
            if not message or label not in {"0", "1"}:
                raise ValueError(f"Invalid message or label at {path}:{number}")
            rows.append((message, int(label)))
        return rows


def unique(rows, name):
    result = {}
    for message, label in rows:
        normalized = key(message)
        if normalized in result and result[normalized][1] != label:
            raise ValueError(f"Conflicting labels in {name}: {message[:60]}")
        result.setdefault(normalized, (message, label))
    return result


def prepare(train_path, test_path, validation_fraction, seed):
    train_all = unique(read_csv(train_path), str(train_path))
    test = unique(read_csv(test_path), str(test_path))
    conflicts = [k for k in train_all.keys() & test.keys() if train_all[k][1] != test[k][1]]
    if conflicts:
        raise ValueError(f"{len(conflicts)} train/test messages have conflicting labels")
    overlap = len(train_all.keys() & test.keys())
    pool = [row for k, row in train_all.items() if k not in test]
    by_label = {label: [row for row in pool if row[1] == label] for label in (0, 1)}
    rng = random.Random(seed)
    training, validation = [], []
    for label, group in by_label.items():
        if len(group) < 2:
            raise ValueError(f"Not enough class {label} messages for a validation split")
        rng.shuffle(group)
        count = max(1, round(len(group) * validation_fraction))
        validation.extend(group[:count])
        training.extend(group[count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation, list(test.values()), overlap


def metrics(labels, probabilities, low, high):
    predictions = [int(p >= 0.5) for p in probabilities]
    scores = {}
    for cls in (0, 1):
        tp = sum(y == cls and p == cls for y, p in zip(labels, predictions))
        fp = sum(y != cls and p == cls for y, p in zip(labels, predictions))
        fn = sum(y == cls and p != cls for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores["ham" if cls == 0 else "scam"] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(y == cls for y in labels)}
    scores["macro_f1"] = (scores["ham"]["f1"] + scores["scam"]["f1"]) / 2
    scores["accuracy"] = sum(y == p for y, p in zip(labels, predictions)) / len(labels)
    scores["false_negative_count"] = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    scores["review_fraction"] = sum(low <= p <= high for p in probabilities) / len(probabilities)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data/raw/vietnamese_sms_merged_all.csv")
    parser.add_argument("--test-csv", type=Path, default=ROOT / "data/raw/vietnamese_sms_test.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/visobert_finetuned")
    parser.add_argument("--checkpoint", default="uitnlp/visobert")
    parser.add_argument("--revision", default="909aedfadee606de2ab6239b74f771df55a1a7cb",
                        help="Official ViSoBERT revision containing model.safetensors")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--scam-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-low", type=float, default=0.35)
    parser.add_argument("--threshold-high", type=float, default=0.85)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1 or not 0 < args.threshold_low < args.threshold_high < 1:
        parser.error("validation fraction and thresholds must be ordered within (0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.scam_weight <= 0:
        parser.error("epochs, batch size, and scam weight must be positive")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    training, validation, test, overlap = prepare(args.train_csv, args.test_csv, args.validation_fraction, args.seed)
    print(json.dumps({"train": len(training), "validation": len(validation), "test": len(test), "test_removed_from_train": overlap,
                      "train_labels": dict(Counter(label for _, label in training))}, indent=2))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, revision=args.revision, use_safetensors=True,
        num_labels=2, id2label={0: "HAM", 1: "SCAM"}, label2id={"HAM": 0, "SCAM": 1}
    ).to(device)
    if device.type == "cuda":
        model.gradient_checkpointing_enable()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    class Messages(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    def collate(rows):
        encoded = tokenizer([message for message, _ in rows], padding=True, truncation=True,
                            max_length=args.max_length, return_tensors="pt")
        encoded["labels"] = torch.tensor([label for _, label in rows], dtype=torch.long)
        return encoded

    train_loader = DataLoader(Messages(training), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(Messages(validation), batch_size=args.batch_size, collate_fn=collate)
    test_loader = DataLoader(Messages(test), batch_size=args.batch_size, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    weights = torch.tensor([1.0, args.scam_weight], device=device)

    def predict(loader):
        model.eval()
        all_logits, all_labels = [], []
        with torch.inference_mode():
            for batch in loader:
                labels = batch.pop("labels")
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                all_logits.append(logits.cpu())
                all_labels.append(labels)
        return torch.cat(all_logits), torch.cat(all_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        started = time.perf_counter()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
                loss = F.cross_entropy(logits, labels, weight=weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        logits, labels = predict(valid_loader)
        probs = logits.softmax(dim=-1)[:, 1].tolist()
        result = metrics(labels.tolist(), probs, args.threshold_low, args.threshold_high)
        entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "validation": result,
                 "seconds": round(time.perf_counter() - started, 1)}
        history.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
        if result["macro_f1"] > best_f1:
            best_f1 = result["macro_f1"]
            model.save_pretrained(args.output_dir, safe_serialization=True)
            tokenizer.save_pretrained(args.output_dir)

    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    validation_logits, validation_labels = predict(valid_loader)
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    calibrator = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        calibrator.zero_grad()
        loss = F.cross_entropy(validation_logits / log_temperature.exp(), validation_labels)
        loss.backward()
        return loss

    calibrator.step(closure)
    temperature = float(log_temperature.exp().detach())
    test_logits, test_labels = predict(test_loader)
    test_probs = (test_logits / temperature).softmax(dim=-1)[:, 1].tolist()
    report = {
        "base_checkpoint": args.checkpoint,
        "base_revision": args.revision,
        "training_csv_sha256": hashlib.sha256(args.train_csv.read_bytes()).hexdigest(),
        "test_csv_sha256": hashlib.sha256(args.test_csv.read_bytes()).hexdigest(),
        "seed": args.seed, "train_count": len(training), "validation_count": len(validation),
        "test_count": len(test), "test_duplicates_excluded_from_training": overlap,
        "temperature": temperature, "scam_weight": args.scam_weight,
        "threshold_low": args.threshold_low, "threshold_high": args.threshold_high,
        "max_length": args.max_length, "device": str(device), "history": history,
        "test_metrics": metrics(test_labels.tolist(), test_probs, args.threshold_low, args.threshold_high),
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "test_metrics": report["test_metrics"],
                      "temperature": temperature}, indent=2))


if __name__ == "__main__":
    main()
