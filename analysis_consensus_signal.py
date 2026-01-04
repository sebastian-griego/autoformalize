#!/usr/bin/env python3
"""Analyze which candidate-set properties predict consensus gains."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple


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


def _pairwise_avg(tokens: List[Counter[str]]) -> float:
    if len(tokens) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            total += _f1_score(tokens[i], tokens[j])
            count += 1
    return total / count if count else 0.0


def _consensus_scores(tokens: List[Counter[str]]) -> List[float]:
    if len(tokens) <= 1:
        return [0.0 for _ in tokens]
    scores: List[float] = []
    for i, tok in enumerate(tokens):
        acc = 0.0
        for j, other in enumerate(tokens):
            if i == j:
                continue
            acc += _f1_score(tok, other)
        scores.append(acc / (len(tokens) - 1))
    return scores


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze consensus improvement signals.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Best-of-cycle JSON files")
    parser.add_argument("--dedupe-id", action="store_true", help="Deduplicate records by example id")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = []
    seen_ids = set()
    for path in args.results:
        data = json.loads(path.read_text())
        for record in data.get("records", []):
            if not isinstance(record, dict):
                continue
            if args.dedupe_id:
                rec_id = record.get("id")
                if rec_id in seen_ids:
                    continue
                if rec_id is not None:
                    seen_ids.add(rec_id)
            records.append(record)

    improvements: List[float] = []
    oracle_improvements: List[float] = []
    counts: List[float] = []
    tc_rates: List[float] = []
    avg_pairwise: List[float] = []
    avg_pairwise_tc: List[float] = []
    consensus_margins: List[float] = []
    baseline_f1s: List[float] = []

    baseline_sum = 0.0
    consensus_sum = 0.0
    oracle_sum = 0.0
    improve_count = 0
    total = 0

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

        texts: List[str] = []
        tokens: List[Counter[str]] = []
        typechecks: List[bool] = []
        f1s: List[float] = []
        for entry in entries:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            texts.append(candidate)
            tok = _counter(_tokenize(_normalize_statement(candidate)))
            tokens.append(tok)
            typechecks.append(bool(entry.get("typecheck", False)))
            f1s.append(_statement_f1(candidate, ground_truth))

        if not texts:
            continue

        baseline_f1 = _statement_f1(baseline_candidate, ground_truth)
        consensus_scores = _consensus_scores(tokens)
        best_idx = max(range(len(texts)), key=lambda i: consensus_scores[i])
        consensus_f1 = f1s[best_idx]
        oracle_f1 = max(f1s)

        baseline_sum += baseline_f1
        consensus_sum += consensus_f1
        oracle_sum += oracle_f1
        total += 1
        if consensus_f1 > baseline_f1:
            improve_count += 1

        improvements.append(consensus_f1 - baseline_f1)
        oracle_improvements.append(oracle_f1 - baseline_f1)
        counts.append(float(len(texts)))
        tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0
        tc_rates.append(tc_rate)
        avg_pairwise.append(_pairwise_avg(tokens))
        tc_tokens = [tok for tok, tc in zip(tokens, typechecks) if tc]
        avg_pairwise_tc.append(_pairwise_avg(tc_tokens) if len(tc_tokens) >= 2 else 0.0)
        baseline_f1s.append(baseline_f1)
        sorted_scores = sorted(consensus_scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        consensus_margins.append(margin)

    summary = {
        "count": total,
        "baseline_f1_avg": baseline_sum / total if total else 0.0,
        "consensus_f1_avg": consensus_sum / total if total else 0.0,
        "oracle_f1_avg": oracle_sum / total if total else 0.0,
        "consensus_improve_rate": improve_count / total if total else 0.0,
        "correlations": {
            "improve_vs_count": _pearson(improvements, counts),
            "improve_vs_tc_rate": _pearson(improvements, tc_rates),
            "improve_vs_pairwise": _pearson(improvements, avg_pairwise),
            "improve_vs_pairwise_tc": _pearson(improvements, avg_pairwise_tc),
            "improve_vs_consensus_margin": _pearson(improvements, consensus_margins),
            "improve_vs_baseline_f1": _pearson(improvements, baseline_f1s),
            "oracle_improve_vs_count": _pearson(oracle_improvements, counts),
            "oracle_improve_vs_pairwise": _pearson(oracle_improvements, avg_pairwise),
        },
    }

    if args.output is not None:
        payload = {
            "meta": {"results_paths": [str(p) for p in args.results], "dedupe_id": args.dedupe_id},
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(key + ":")
            for sub_key, sub_val in value.items():
                print(f"  {sub_key}: {sub_val:.6f}")
        else:
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
