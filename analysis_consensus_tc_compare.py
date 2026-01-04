#!/usr/bin/env python3
"""Compare consensus over all candidates vs typechecked-only candidates."""

from __future__ import annotations

import argparse
import json
import math
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


def _consensus_index(tokens: List[Counter[str]]) -> int:
    if len(tokens) == 1:
        return 0
    best_idx = 0
    best_score = float("-inf")
    for i, tok in enumerate(tokens):
        score = 0.0
        for j, other in enumerate(tokens):
            if i == j:
                continue
            score += _f1_score(tok, other)
        score /= max(1, len(tokens) - 1)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


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
    parser = argparse.ArgumentParser(description="Compare all-candidate vs tc-only consensus.")
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

    diffs: List[float] = []
    tc_rates: List[float] = []
    tc_counts: List[float] = []
    all_f1_sum = 0.0
    tc_f1_sum = 0.0
    count = 0
    better_tc = 0
    both = 0

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

        texts = []
        tokens = []
        typechecks = []
        f1s = []
        for entry in entries:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            texts.append(candidate)
            tokens.append(_counter(_tokenize(_normalize_statement(candidate))))
            typechecks.append(bool(entry.get("typecheck", False)))
            f1s.append(_statement_f1(candidate, ground_truth))

        if not texts:
            continue

        all_idx = _consensus_index(tokens)
        all_f1 = f1s[all_idx]
        tc_tokens = [tok for tok, tc in zip(tokens, typechecks) if tc]
        tc_f1 = all_f1
        if tc_tokens:
            tc_indices = [i for i, tc in enumerate(typechecks) if tc]
            tc_idx_local = _consensus_index(tc_tokens)
            tc_f1 = f1s[tc_indices[tc_idx_local]]

        if tc_tokens:
            both += 1
            if tc_f1 > all_f1:
                better_tc += 1

        tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0
        diffs.append(tc_f1 - all_f1)
        tc_rates.append(tc_rate)
        tc_counts.append(sum(typechecks))
        all_f1_sum += all_f1
        tc_f1_sum += tc_f1
        count += 1

    summary = {
        "count": count,
        "all_consensus_f1_avg": all_f1_sum / count if count else 0.0,
        "tc_consensus_f1_avg": tc_f1_sum / count if count else 0.0,
        "tc_better_rate": better_tc / both if both else 0.0,
        "diff_vs_tc_rate": _pearson(diffs, tc_rates),
        "diff_vs_tc_count": _pearson(diffs, tc_counts),
    }

    if args.output is not None:
        payload = {
            "meta": {"results_paths": [str(p) for p in args.results], "dedupe_id": args.dedupe_id},
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
