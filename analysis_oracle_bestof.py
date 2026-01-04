#!/usr/bin/env python3
"""Compute oracle best-of metrics for candidate sets."""

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
    parser = argparse.ArgumentParser(description="Compute oracle best-of metrics.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "oracle_f1_sum": 0.0,
        "oracle_tc_f1_sum": 0.0,
        "oracle_typecheck_sum": 0.0,
        "oracle_tc_typecheck_sum": 0.0,
        "oracle_improve_sum": 0,
        "oracle_tc_improve_sum": 0,
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

        baseline_f1 = _statement_f1(baseline_candidate, ground_truth)

        best_all_f1 = -1.0
        best_all_typecheck = False
        best_tc_f1 = -1.0
        best_tc_typecheck = False
        for entry in entries:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str):
                continue
            f1 = _statement_f1(candidate, ground_truth)
            is_tc = bool(entry.get("typecheck", False))
            if f1 > best_all_f1:
                best_all_f1 = f1
                best_all_typecheck = is_tc
            if is_tc and f1 > best_tc_f1:
                best_tc_f1 = f1
                best_tc_typecheck = True

        if best_all_f1 < 0:
            continue
        if best_tc_f1 < 0:
            best_tc_f1 = best_all_f1
            best_tc_typecheck = best_all_typecheck

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += baseline_f1
        metrics["oracle_f1_sum"] += best_all_f1
        metrics["oracle_tc_f1_sum"] += best_tc_f1
        metrics["oracle_typecheck_sum"] += int(best_all_typecheck)
        metrics["oracle_tc_typecheck_sum"] += int(best_tc_typecheck)
        metrics["oracle_improve_sum"] += int(best_all_f1 > baseline_f1)
        metrics["oracle_tc_improve_sum"] += int(best_tc_f1 > baseline_f1)

    summary = {
        "count": metrics["count"],
        "baseline_f1_avg": metrics["baseline_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_f1_avg": metrics["oracle_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_tc_f1_avg": metrics["oracle_tc_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_typecheck_rate": metrics["oracle_typecheck_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_tc_typecheck_rate": metrics["oracle_tc_typecheck_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_improve_rate": metrics["oracle_improve_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "oracle_tc_improve_rate": metrics["oracle_tc_improve_sum"] / metrics["count"] if metrics["count"] else 0.0,
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
