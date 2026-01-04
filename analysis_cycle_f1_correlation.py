#!/usr/bin/env python3
"""Analyze correlation between cycle scores and statement similarity."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
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


def _rankdata(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


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


def _spearman(xs: List[float], ys: List[float]) -> float:
    return _pearson(_rankdata(xs), _rankdata(ys))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-example cycle/F1 correlations.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    pearsons: List[float] = []
    spearmans: List[float] = []
    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or len(candidates) < 2:
            continue
        ground_truth = record.get("ground_truth", "")
        scores: List[float] = []
        f1s: List[float] = []
        for entry in candidates:
            cycle = entry.get("cycle_score")
            candidate = entry.get("candidate")
            if not isinstance(cycle, (int, float)) or not isinstance(candidate, str):
                continue
            if not math.isfinite(float(cycle)):
                continue
            scores.append(float(cycle))
            f1s.append(_statement_f1(candidate, ground_truth))
        if len(scores) < 2:
            continue
        pearsons.append(_pearson(scores, f1s))
        spearmans.append(_spearman(scores, f1s))

    summary = {
        "count": len(pearsons),
        "pearson_avg": sum(pearsons) / len(pearsons) if pearsons else 0.0,
        "pearson_median": median(pearsons) if pearsons else 0.0,
        "spearman_avg": sum(spearmans) / len(spearmans) if spearmans else 0.0,
        "spearman_median": median(spearmans) if spearmans else 0.0,
    }

    if args.output is not None:
        payload = {"meta": {"results_path": str(args.results)}, "summary": summary}
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
