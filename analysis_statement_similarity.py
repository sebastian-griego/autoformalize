#!/usr/bin/env python3
"""Compute token-level similarity between generated and ground-truth statements."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")


def _extract_theorem_block(text: str) -> str:
    if not text:
        return ""
    lower = text.lower()
    idx = lower.rfind("theorem")
    if idx == -1:
        return text.strip()
    snippet = text[idx:].strip()
    head, sep, _ = snippet.partition(":=")
    if sep:
        snippet = head.strip()
    return snippet


def _strip_theorem_name(text: str) -> str:
    if not text:
        return ""
    parts = text.split(None, 2)
    if len(parts) < 2:
        return text.strip()
    if parts[0] != "theorem":
        return text.strip()
    if len(parts) == 2:
        return "theorem"
    return "theorem " + parts[2]


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _counter(tokens: Iterable[str]) -> Counter[str]:
    return Counter(tokens)


def _f1_score(pred: Counter[str], gold: Counter[str]) -> float:
    if not pred or not gold:
        return 0.0
    common = sum((pred & gold).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize_statement(text: str) -> str:
    block = _extract_theorem_block(text)
    return _strip_theorem_name(block)


def _score_pair(pred_text: str, gold_text: str) -> float:
    pred = _counter(_tokenize(_normalize_statement(pred_text)))
    gold = _counter(_tokenize(_normalize_statement(gold_text)))
    return _f1_score(pred, gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute token-level F1 similarity to ground truth.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    summary = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "best_cycle_f1_sum": 0.0,
        "best_cycle_tc_f1_sum": 0.0,
        "best_cycle_improve": 0,
        "best_cycle_tc_improve": 0,
        "best_lenpen_count": 0,
        "best_lenpen_f1_sum": 0.0,
        "best_lenpen_tc_count": 0,
        "best_lenpen_tc_f1_sum": 0.0,
        "best_lenpen_improve": 0,
        "best_lenpen_tc_improve": 0,
    }

    for record in records:
        ground_truth = record.get("ground_truth", "")
        baseline = record.get("baseline_candidate", "")
        best_cycle = record.get("best_cycle_candidate", "")
        best_cycle_tc = record.get("best_cycle_tc_candidate", best_cycle)

        baseline_f1 = _score_pair(baseline, ground_truth)
        best_cycle_f1 = _score_pair(best_cycle, ground_truth)
        best_cycle_tc_f1 = _score_pair(best_cycle_tc, ground_truth)

        record["baseline_stmt_f1"] = baseline_f1
        record["best_cycle_stmt_f1"] = best_cycle_f1
        record["best_cycle_tc_stmt_f1"] = best_cycle_tc_f1

        best_lenpen = record.get("best_lenpen_candidate")
        if best_lenpen is not None:
            best_lenpen_f1 = _score_pair(str(best_lenpen), ground_truth)
            record["best_lenpen_stmt_f1"] = best_lenpen_f1
            summary["best_lenpen_count"] += 1
            summary["best_lenpen_f1_sum"] += best_lenpen_f1
            if best_lenpen_f1 > baseline_f1:
                summary["best_lenpen_improve"] += 1

        best_lenpen_tc = record.get("best_lenpen_tc_candidate")
        if best_lenpen_tc is not None:
            best_lenpen_tc_f1 = _score_pair(str(best_lenpen_tc), ground_truth)
            record["best_lenpen_tc_stmt_f1"] = best_lenpen_tc_f1
            summary["best_lenpen_tc_count"] += 1
            summary["best_lenpen_tc_f1_sum"] += best_lenpen_tc_f1
            if best_lenpen_tc_f1 > baseline_f1:
                summary["best_lenpen_tc_improve"] += 1

        summary["count"] += 1
        summary["baseline_f1_sum"] += baseline_f1
        summary["best_cycle_f1_sum"] += best_cycle_f1
        summary["best_cycle_tc_f1_sum"] += best_cycle_tc_f1
        if best_cycle_f1 > baseline_f1:
            summary["best_cycle_improve"] += 1
        if best_cycle_tc_f1 > baseline_f1:
            summary["best_cycle_tc_improve"] += 1

    out_summary = {
        "count": summary["count"],
        "baseline_f1_avg": summary["baseline_f1_sum"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_f1_avg": summary["best_cycle_f1_sum"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_tc_f1_avg": summary["best_cycle_tc_f1_sum"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_improve_rate": summary["best_cycle_improve"] / summary["count"] if summary["count"] else 0.0,
        "best_cycle_tc_improve_rate": summary["best_cycle_tc_improve"] / summary["count"] if summary["count"] else 0.0,
    }
    if summary["best_lenpen_count"]:
        out_summary["best_lenpen_f1_avg"] = summary["best_lenpen_f1_sum"] / summary["best_lenpen_count"]
        out_summary["best_lenpen_improve_rate"] = summary["best_lenpen_improve"] / summary["best_lenpen_count"]
    if summary["best_lenpen_tc_count"]:
        out_summary["best_lenpen_tc_f1_avg"] = summary["best_lenpen_tc_f1_sum"] / summary["best_lenpen_tc_count"]
        out_summary["best_lenpen_tc_improve_rate"] = summary["best_lenpen_tc_improve"] / summary["best_lenpen_tc_count"]

    output_payload: Dict[str, object] = {
        "meta": {"results_path": str(args.results)},
        "summary": out_summary,
        "records": records,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(output_payload, indent=2))

    print("Summary:")
    for key, value in out_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
