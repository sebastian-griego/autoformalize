#!/usr/bin/env python3
"""Estimate consensus F1 as a function of candidate subset size."""

from __future__ import annotations

import argparse
import json
import random
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


def _prepare_candidates(record: dict, *, require_typecheck: bool) -> Tuple[List[str], List[bool], List[Counter[str]]]:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return [], [], []
    texts = []
    tcs = []
    tokens = []
    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        tc = bool(entry.get("typecheck", False))
        if require_typecheck and not tc:
            continue
        texts.append(candidate)
        tcs.append(tc)
        tokens.append(_counter(_tokenize(_normalize_statement(candidate))))
    return texts, tcs, tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Consensus F1 vs subset size.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--ks", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    summary = {}
    for require_tc in (False, True):
        key = "tc_only" if require_tc else "all"
        results = {}
        for k in args.ks:
            count = 0
            f1_sum = 0.0
            tc_sum = 0.0
            for record in records:
                texts, tcs, tokens = _prepare_candidates(record, require_typecheck=require_tc)
                if len(texts) < k:
                    continue
                ground_truth = record.get("ground_truth", "")
                for _ in range(args.trials):
                    subset_indices = rng.sample(range(len(texts)), k)
                    subset_tokens = [tokens[i] for i in subset_indices]
                    pick_idx = _consensus_index(subset_tokens)
                    chosen = texts[subset_indices[pick_idx]]
                    chosen_tc = tcs[subset_indices[pick_idx]]
                    f1_sum += _statement_f1(chosen, ground_truth)
                    tc_sum += int(bool(chosen_tc))
                    count += 1
            results[f"k_{k}"] = {
                "count": count,
                "f1_avg": f1_sum / count if count else 0.0,
                "typecheck_rate": tc_sum / count if count else 0.0,
            }
        summary[key] = results

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results),
                "ks": args.ks,
                "trials": args.trials,
                "seed": args.seed,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for group, results in summary.items():
        print(group + ":")
        for k_key, metrics in results.items():
            print(
                f"  {k_key} f1={metrics['f1_avg']:.3f} "
                f"tc={metrics['typecheck_rate']:.3f} "
                f"count={metrics['count']}"
            )


if __name__ == "__main__":
    main()
