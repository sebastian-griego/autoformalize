#!/usr/bin/env python3
"""Measure how typechecking correlates with statement similarity."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List


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


def _normalize_statement(text: str) -> str:
    block = _extract_theorem_block(text)
    return _strip_theorem_name(block)


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


def _statement_f1(pred_text: str, gold_text: str) -> float:
    pred = _counter(_tokenize(_normalize_statement(pred_text)))
    gold = _counter(_tokenize(_normalize_statement(gold_text)))
    return _f1_score(pred, gold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze typecheck signal strength.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "tc_count": 0,
        "non_tc_count": 0,
        "tc_f1_sum": 0.0,
        "non_tc_f1_sum": 0.0,
        "max_tc_f1_sum": 0.0,
        "max_non_tc_f1_sum": 0.0,
        "records_with_tc": 0,
        "records_with_non_tc": 0,
        "records_with_both": 0,
    }

    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        ground_truth = record.get("ground_truth", "")
        baseline_candidate = record.get("baseline_candidate", "")
        baseline_typecheck = bool(record.get("baseline_typecheck", False))

        entries = list(candidates)
        if baseline_candidate and not any(
            isinstance(entry, dict) and entry.get("candidate") == baseline_candidate
            for entry in entries
        ):
            entries.append({"candidate": baseline_candidate, "typecheck": baseline_typecheck})

        metrics["count"] += 1
        max_tc = None
        max_non_tc = None
        for entry in entries:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str):
                continue
            f1 = _statement_f1(candidate, ground_truth)
            if entry.get("typecheck", False):
                metrics["tc_count"] += 1
                metrics["tc_f1_sum"] += f1
                max_tc = f1 if max_tc is None else max(max_tc, f1)
            else:
                metrics["non_tc_count"] += 1
                metrics["non_tc_f1_sum"] += f1
                max_non_tc = f1 if max_non_tc is None else max(max_non_tc, f1)

        if max_tc is not None:
            metrics["records_with_tc"] += 1
            metrics["max_tc_f1_sum"] += max_tc
        if max_non_tc is not None:
            metrics["records_with_non_tc"] += 1
            metrics["max_non_tc_f1_sum"] += max_non_tc
        if max_tc is not None and max_non_tc is not None:
            metrics["records_with_both"] += 1

    summary = {
        "records": metrics["count"],
        "candidate_typecheck_rate": metrics["tc_count"]
        / (metrics["tc_count"] + metrics["non_tc_count"])
        if (metrics["tc_count"] + metrics["non_tc_count"])
        else 0.0,
        "tc_candidate_f1_avg": metrics["tc_f1_sum"] / metrics["tc_count"] if metrics["tc_count"] else 0.0,
        "non_tc_candidate_f1_avg": metrics["non_tc_f1_sum"] / metrics["non_tc_count"] if metrics["non_tc_count"] else 0.0,
        "max_tc_f1_avg": metrics["max_tc_f1_sum"] / metrics["records_with_tc"]
        if metrics["records_with_tc"]
        else 0.0,
        "max_non_tc_f1_avg": metrics["max_non_tc_f1_sum"] / metrics["records_with_non_tc"]
        if metrics["records_with_non_tc"]
        else 0.0,
        "records_with_tc": metrics["records_with_tc"],
        "records_with_non_tc": metrics["records_with_non_tc"],
        "records_with_both": metrics["records_with_both"],
    }

    if args.output is not None:
        payload = {"meta": {"results_path": str(args.results)}, "summary": summary}
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
