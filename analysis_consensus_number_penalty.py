#!/usr/bin/env python3
"""Consensus rerank with NL-number penalties."""

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


def _consensus_scores(token_counters: List[Counter[str]]) -> List[float]:
    if len(token_counters) == 1:
        return [0.0]
    scores = []
    for i, tok in enumerate(token_counters):
        total = 0.0
        for j, other in enumerate(token_counters):
            if i == j:
                continue
            total += _f1_score(tok, other)
        scores.append(total / max(1, len(token_counters) - 1))
    return scores


def _nl_number_counter(text: str) -> Counter[str]:
    digits = NUM_RE.findall(text)
    words = [WORD_NUMBERS[word.lower()] for word in WORD_NUM_RE.findall(text)]
    return Counter(digits + words)


def _missing_extra_ratio(candidate: str, nl_nums: Counter[str]) -> Tuple[float, float]:
    if not nl_nums:
        return 0.0, 0.0
    cand_nums = Counter(NUM_RE.findall(_normalize_statement(candidate)))
    common = sum((cand_nums & nl_nums).values())
    cand_total = sum(cand_nums.values())
    nl_total = sum(nl_nums.values())
    missing_ratio = float(nl_total - common) / float(nl_total) if nl_total else 0.0
    extra_ratio = float(cand_total - common) / float(nl_total) if nl_total else 0.0
    return missing_ratio, extra_ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="Consensus rerank with NL-number penalties.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    summary = {}
    for alpha in args.alphas:
        for beta in args.betas:
            count = 0
            f1_sum = 0.0
            digit_count = 0
            digit_sum = 0.0
            nodigits_count = 0
            nodigits_sum = 0.0
            for record in records:
                candidates = record.get("candidates", [])
                if not isinstance(candidates, list) or not candidates:
                    continue
                ground_truth = record.get("ground_truth", "")
                nl_statement = record.get("nl_statement", "") or ""
                nl_nums = _nl_number_counter(nl_statement)

                token_counters = []
                cand_texts = []
                for entry in candidates:
                    cand = entry.get("candidate")
                    if not isinstance(cand, str) or not cand.strip():
                        continue
                    cand_texts.append(cand)
                    token_counters.append(_counter(_tokenize(_normalize_statement(cand))))

                if not cand_texts:
                    continue
                cons_scores = _consensus_scores(token_counters)

                best_idx = 0
                best_score = float("-inf")
                for idx, cand in enumerate(cand_texts):
                    missing_ratio, extra_ratio = _missing_extra_ratio(cand, nl_nums)
                    score = cons_scores[idx] - alpha * missing_ratio - beta * extra_ratio
                    if score > best_score:
                        best_score = score
                        best_idx = idx

                chosen = cand_texts[best_idx]
                f1 = _statement_f1(chosen, ground_truth)
                f1_sum += f1
                count += 1
                if nl_nums:
                    digit_count += 1
                    digit_sum += f1
                else:
                    nodigits_count += 1
                    nodigits_sum += f1

            summary[f"alpha_{alpha}_beta_{beta}"] = {
                "count": count,
                "f1_avg": f1_sum / count if count else 0.0,
                "digits_count": digit_count,
                "digits_f1_avg": digit_sum / digit_count if digit_count else 0.0,
                "nodigits_count": nodigits_count,
                "nodigits_f1_avg": nodigits_sum / nodigits_count if nodigits_count else 0.0,
            }

    if args.output is not None:
        payload = {
            "meta": {"results_path": str(args.results), "alphas": args.alphas, "betas": args.betas},
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    best = sorted(summary.items(), key=lambda item: item[1]["f1_avg"], reverse=True)[:10]
    for key, metrics in best:
        print(
            f"{key} f1={metrics['f1_avg']:.3f} "
            f"digits={metrics['digits_f1_avg']:.3f} "
            f"nodigits={metrics['nodigits_f1_avg']:.3f} "
            f"count={metrics['count']}"
        )


if __name__ == "__main__":
    main()
