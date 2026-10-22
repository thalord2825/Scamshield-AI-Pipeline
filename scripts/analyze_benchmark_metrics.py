"""
==============================================================================
ScamShield-VN: Benchmark Telemetry Analysis & Report Generator
==============================================================================
Author: ScamShield Research & Evaluation Team
Description:
    Analyzes telemetry from SQLite / JSONL and computes:
    - Precision, Recall, Specificity, False Positive Rate (FPR)
    - Accuracy and Macro-F1
    - Fast-Path Exit Ratio and Escalation Rate
    - Latency statistics (P50, P90, P99, Mean)
    - Total API Calls and Cost per 1,000 messages
    - Generates markdown comparison tables and JSON reports.
==============================================================================
"""

import os
import sys
import json
import sqlite3
import argparse
from typing import Dict, List, Any
import numpy as np

# Ensure UTF-8 output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Two-Tier Benchmark Telemetry")
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.path.join(ROOT_DIR, "experiments", "benchmark_results.db"),
        help="SQLite database path",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run_id to filter (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results"),
        help="Directory to save generated analysis reports",
    )
    return parser.parse_args()


def compute_metrics_for_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {}

    tp = fp = tn = fn = unknowns = 0
    latencies = []
    costs = []
    calls = []
    fastpath_count = 0
    escalated_count = 0
    defender_count = 0

    for r in rows:
        gold = r["gold_label"]
        pred = r["predicted_label"]
        lat = r["latency_ms"]
        cost = r["cost_usd"]
        call_cnt = r["api_calls_count"]

        latencies.append(lat)
        costs.append(cost)
        calls.append(call_cnt)

        if r["escalated"]:
            escalated_count += 1
        else:
            fastpath_count += 1

        if r["defender_intervened"]:
            defender_count += 1

        if pred == "UNKNOWN":
            unknowns += 1
            # In strict binary scoring, unknown is counted as non-positive for scam
            if gold == "SCAM":
                fn += 1
            else:
                tn += 1
        elif gold == "SCAM" and pred == "SCAM":
            tp += 1
        elif gold == "HAM" and pred == "SCAM":
            fp += 1
        elif gold == "HAM" and pred == "HAM":
            tn += 1
        elif gold == "SCAM" and pred == "HAM":
            fn += 1

    accuracy = ((tp + tn) / total) * 100.0 if total else 0.0
    precision_scam = (tp / (tp + fp)) * 100.0 if (tp + fp) else 0.0
    recall_scam = (tp / (tp + fn)) * 100.0 if (tp + fn) else 0.0
    f1_scam = (2 * precision_scam * recall_scam) / (precision_scam + recall_scam) if (precision_scam + recall_scam) else 0.0

    precision_ham = (tn / (tn + fn)) * 100.0 if (tn + fn) else 0.0
    recall_ham = (tn / (tn + fp)) * 100.0 if (tn + fp) else 0.0
    f1_ham = (2 * precision_ham * recall_ham) / (precision_ham + recall_ham) if (precision_ham + recall_ham) else 0.0

    macro_f1 = (f1_scam + f1_ham) / 2.0
    fpr = (fp / (fp + tn)) * 100.0 if (fp + tn) else 0.0

    fastpath_pct = (fastpath_count / total) * 100.0
    escalation_pct = (escalated_count / total) * 100.0

    lat_arr = np.array(latencies)
    cost_total = sum(costs)
    cost_per_1k = (cost_total / total) * 1000.0 if total else 0.0

    return {
        "total_samples": total,
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "UNKNOWN": unknowns,
        },
        "accuracy": round(accuracy, 2),
        "precision_scam": round(precision_scam, 2),
        "recall_scam": round(recall_scam, 2),
        "f1_scam": round(f1_scam, 2),
        "macro_f1": round(macro_f1, 2),
        "fpr": round(fpr, 2),
        "fastpath_pct": round(fastpath_pct, 2),
        "escalation_pct": round(escalation_pct, 2),
        "defender_interventions": defender_count,
        "total_api_calls": sum(calls),
        "avg_calls_per_sample": round(sum(calls) / total, 2),
        "total_cost_usd": round(cost_total, 5),
        "cost_per_1k_usd": round(cost_per_1k, 4),
        "latency": {
            "mean": round(float(np.mean(lat_arr)), 1),
            "p50": round(float(np.percentile(lat_arr, 50)), 1),
            "p90": round(float(np.percentile(lat_arr, 90)), 1),
            "p99": round(float(np.percentile(lat_arr, 99)), 1),
        },
    }


def analyze_db():
    args = parse_args()
    if not os.path.exists(args.db_path):
        print(f"Error: Database file '{args.db_path}' does not exist.")
        return

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if args.run_id:
        cur.execute("SELECT DISTINCT mode FROM sample_telemetry WHERE run_id = ?", (args.run_id,))
    else:
        cur.execute("SELECT DISTINCT mode FROM sample_telemetry")

    modes = [r["mode"] for r in cur.fetchall()]

    summary = {}
    for m in modes:
        if args.run_id:
            cur.execute("SELECT * FROM sample_telemetry WHERE mode = ? AND run_id = ?", (m, args.run_id))
        else:
            cur.execute("SELECT * FROM sample_telemetry WHERE mode = ?", (m,))
        rows = [dict(r) for r in cur.fetchall()]
        summary[m] = compute_metrics_for_rows(rows)

    conn.close()

    # Generate Markdown Table
    md_output = "# ScamShield Benchmark Evaluation Report\n\n"
    md_output += "| Metric / Evaluation Arm | Solo Tier 1 (ViSoBERT) | Solo Tier 2 (3-Agent) | Cascaded Two-Tier |\n"
    md_output += "| :--- | :---: | :---: | :---: |\n"

    def get_val(mode_key, key, default="N/A"):
        m = summary.get(mode_key, {})
        val = m.get(key, default)
        return f"{val}"

    md_output += f"| **Total Samples** | {get_val('tier1_only', 'total_samples')} | {get_val('tier2_only', 'total_samples')} | {get_val('cascaded', 'total_samples')} |\n"
    md_output += f"| **Accuracy (%)** | {get_val('tier1_only', 'accuracy')}% | {get_val('tier2_only', 'accuracy')}% | {get_val('cascaded', 'accuracy')}% |\n"
    md_output += f"| **Macro-F1 (%)** | {get_val('tier1_only', 'macro_f1')}% | {get_val('tier2_only', 'macro_f1')}% | {get_val('cascaded', 'macro_f1')}% |\n"
    md_output += f"| **Scam Recall (%)** | {get_val('tier1_only', 'recall_scam')}% | {get_val('tier2_only', 'recall_scam')}% | {get_val('cascaded', 'recall_scam')}% |\n"
    md_output += f"| **False Positive Rate (FPR)** | {get_val('tier1_only', 'fpr')}% | {get_val('tier2_only', 'fpr')}% | {get_val('cascaded', 'fpr')}% |\n"
    md_output += f"| **Fast-Path Exit Ratio** | {get_val('tier1_only', 'fastpath_pct')}% | 0.00% | {get_val('cascaded', 'fastpath_pct')}% |\n"
    md_output += f"| **Escalation Rate to T2** | 0.00% | 100.00% | {get_val('cascaded', 'escalation_pct')}% |\n"
    md_output += f"| **Public Defender Interventions** | N/A | {get_val('tier2_only', 'defender_interventions')} | {get_val('cascaded', 'defender_interventions')} |\n"
    md_output += f"| **Avg Latency (ms)** | {summary.get('tier1_only', {}).get('latency', {}).get('mean', 'N/A')} ms | {summary.get('tier2_only', {}).get('latency', {}).get('mean', 'N/A')} ms | {summary.get('cascaded', {}).get('latency', {}).get('mean', 'N/A')} ms |\n"
    md_output += f"| **Cost per 1k Messages ($)** | $0.0000 | ${get_val('tier2_only', 'cost_per_1k_usd')} | ${get_val('cascaded', 'cost_per_1k_usd')} |\n"
    md_output += f"| **Total API Calls** | 0 | {get_val('tier2_only', 'total_api_calls')} | {get_val('cascaded', 'total_api_calls')} |\n"

    print("\n" + md_output)

    os.makedirs(args.output_dir, exist_ok=True)
    report_json_path = os.path.join(args.output_dir, "benchmark_metrics.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    report_md_path = os.path.join(args.output_dir, "benchmark_summary.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_output)

    print(f"Saved analysis reports to:\n- {report_json_path}\n- {report_md_path}")


if __name__ == "__main__":
    analyze_db()
