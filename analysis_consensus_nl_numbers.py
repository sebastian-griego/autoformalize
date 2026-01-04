#!/usr/bin/env python3
"""Consensus rerank with NL-number alignment for numeric statements."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")
NUM_RE = re.compile(r"(?<![A-Za-z_])\d+(?![A-Za-z_])")
WORD_NUMBERS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
    "hundred": "100",
    "thousand": "1000",
}
WORD_NUM_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS.keys()) + r")\b", re.IGNORECASE)


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


def _build_candidate_tokens(candidates: List[Dict[str, object]]) -> List[Tuple[Dict[str, object], Counter[str]]]:
    prepared: List[Tuple[Dict[str, object], Counter[str]]] = []
    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        tokens = _counter(_tokenize(_normalize_statement(candidate)))
        prepared.append((entry, tokens))
    return prepared


def _consensus_candidate(candidates: List[Dict[str, object]]) -> Dict[str, object] | None:
    prepared = _build_candidate_tokens(candidates)
    if not prepared:
        return None
    if len(prepared) == 1:
        return prepared[0][0]
    best = None
    best_score = float("-inf")
    for idx, (entry, tokens) in enumerate(prepared):
        score = 0.0
        for jdx, (_, other_tokens) in enumerate(prepared):
            if idx == jdx:
                continue
            score += _f1_score(tokens, other_tokens)
        score /= max(1, len(prepared) - 1)
        if score > best_score:
            best_score = score
            best = entry
        elif score == best_score and best is not None:
            cycle_score = entry.get("cycle_score")
            best_cycle = best.get("cycle_score")
            if isinstance(cycle_score, (int, float)) and isinstance(best_cycle, (int, float)):
                if math.isfinite(cycle_score) and math.isfinite(best_cycle) and cycle_score > best_cycle:
                    best = entry
    return best


def _nl_number_counter(nl_statement: str) -> Counter[str]:
    if not nl_statement:
        return Counter()
    digits = NUM_RE.findall(nl_statement)
    words = [WORD_NUMBERS[word.lower()] for word in WORD_NUM_RE.findall(nl_statement)]
    return Counter(digits + words)


def _candidate_number_score(candidate: str, nl_nums: Counter[str], metric: str) -> float:
    if not nl_nums:
        return 0.0
    cand_nums = Counter(NUM_RE.findall(_normalize_statement(candidate)))
    common = sum((cand_nums & nl_nums).values())
    cand_total = sum(cand_nums.values())
    nl_total = sum(nl_nums.values())
    if metric == "precision":
        return common / cand_total if cand_total else 0.0
    if metric == "recall":
        return common / nl_total if nl_total else 0.0
    return _f1_score(cand_nums, nl_nums)


def _candidate_missing_extra(candidate: str, nl_nums: Counter[str]) -> Tuple[int, int]:
    if not nl_nums:
        return 0, 0
    cand_nums = Counter(NUM_RE.findall(_normalize_statement(candidate)))
    common = sum((cand_nums & nl_nums).values())
    cand_total = sum(cand_nums.values())
    nl_total = sum(nl_nums.values())
    missing = max(0, nl_total - common)
    extra = max(0, cand_total - common)
    return missing, extra


def _select_with_nl_numbers(
    candidates: List[Dict[str, object]],
    nl_nums: Counter[str],
    metric: str,
    mode: str,
) -> Dict[str, object] | None:
    if not candidates:
        return None
    if not nl_nums:
        return _consensus_candidate(candidates)
    if mode == "perfect":
        perfect = []
        for entry in candidates:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            missing, extra = _candidate_missing_extra(candidate, nl_nums)
            if missing == 0 and extra == 0:
                perfect.append(entry)
        if perfect:
            return _consensus_candidate(perfect)
        return _consensus_candidate(candidates)
    if mode in ("no_missing", "no_extra"):
        filtered = []
        for entry in candidates:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            missing, extra = _candidate_missing_extra(candidate, nl_nums)
            if mode == "no_missing" and missing == 0:
                filtered.append(entry)
            elif mode == "no_extra" and extra == 0:
                filtered.append(entry)
        if filtered:
            return _consensus_candidate(filtered)
        return _consensus_candidate(candidates)
    scored = []
    best_f1 = float("-inf")
    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        score = _candidate_number_score(candidate, nl_nums, metric)
        if score > best_f1:
            best_f1 = score
            scored = [entry]
        elif score == best_f1:
            scored.append(entry)
    if not scored:
        return _consensus_candidate(candidates)
    if len(scored) == 1:
        return scored[0]
    return _consensus_candidate(scored)


def main() -> None:
    parser = argparse.ArgumentParser(description="Consensus rerank with NL-number alignment.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument(
        "--metric",
        choices=["f1", "precision", "recall"],
        default="f1",
        help="NL-number alignment metric",
    )
    parser.add_argument(
        "--mode",
        choices=["filter", "perfect", "no_missing", "no_extra"],
        default="filter",
        help="Selection mode for NL-number alignment",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    metrics = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "consensus_f1_sum": 0.0,
        "nl_number_f1_sum": 0.0,
    }
    per_group = {
        "digits": {"count": 0, "consensus_f1_sum": 0.0, "nl_number_f1_sum": 0.0},
        "nodigits": {"count": 0, "consensus_f1_sum": 0.0, "nl_number_f1_sum": 0.0},
    }

    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        ground_truth = record.get("ground_truth", "")
        baseline_candidate = record.get("baseline_candidate", "")
        nl_statement = record.get("nl_statement", "") or ""
        nl_nums = _nl_number_counter(nl_statement)
        has_digits = bool(nl_nums)

        consensus = _consensus_candidate(candidates)
        nl_pick = _select_with_nl_numbers(candidates, nl_nums, args.metric, args.mode)
        if consensus is None:
            consensus = {"candidate": baseline_candidate}
        if nl_pick is None:
            nl_pick = consensus

        metrics["count"] += 1
        metrics["baseline_f1_sum"] += _statement_f1(str(baseline_candidate), ground_truth)
        metrics["consensus_f1_sum"] += _statement_f1(str(consensus.get("candidate", "")), ground_truth)
        metrics["nl_number_f1_sum"] += _statement_f1(str(nl_pick.get("candidate", "")), ground_truth)

        group = per_group["digits"] if has_digits else per_group["nodigits"]
        group["count"] += 1
        group["consensus_f1_sum"] += _statement_f1(str(consensus.get("candidate", "")), ground_truth)
        group["nl_number_f1_sum"] += _statement_f1(str(nl_pick.get("candidate", "")), ground_truth)

    summary = {
        "count": metrics["count"],
        "baseline_f1_avg": metrics["baseline_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "consensus_f1_avg": metrics["consensus_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "nl_number_f1_avg": metrics["nl_number_f1_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "digits": {
            "count": per_group["digits"]["count"],
            "consensus_f1_avg": per_group["digits"]["consensus_f1_sum"] / per_group["digits"]["count"]
            if per_group["digits"]["count"]
            else 0.0,
            "nl_number_f1_avg": per_group["digits"]["nl_number_f1_sum"] / per_group["digits"]["count"]
            if per_group["digits"]["count"]
            else 0.0,
        },
        "nodigits": {
            "count": per_group["nodigits"]["count"],
            "consensus_f1_avg": per_group["nodigits"]["consensus_f1_sum"] / per_group["nodigits"]["count"]
            if per_group["nodigits"]["count"]
            else 0.0,
            "nl_number_f1_avg": per_group["nodigits"]["nl_number_f1_sum"] / per_group["nodigits"]["count"]
            if per_group["nodigits"]["count"]
            else 0.0,
        },
    }

    if args.output is not None:
        payload = {
            "meta": {"results_path": str(args.results), "metric": args.metric, "mode": args.mode},
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    main()
