#!/usr/bin/env python3
"""Sweep gates between all-consensus and tc-only consensus."""

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


def _select_consensus(prepared: List[Tuple[dict, Counter[str]]]) -> Tuple[dict | None, List[float]]:
    if not prepared:
        return None, []
    if len(prepared) == 1:
        return prepared[0][0], [0.0]
    tokens = [tok for _, tok in prepared]
    scores = _consensus_scores(tokens)
    best = None
    best_score = float("-inf")
    for idx, (entry, _) in enumerate(prepared):
        score = scores[idx]
        if score > best_score:
            best_score = score
            best = entry
        elif score == best_score and best is not None:
            cycle_score = entry.get("cycle_score")
            best_cycle = best.get("cycle_score")
            if isinstance(cycle_score, (int, float)) and isinstance(best_cycle, (int, float)):
                if math.isfinite(cycle_score) and math.isfinite(best_cycle) and cycle_score > best_cycle:
                    best = entry
    return best, scores


def _quantile_thresholds(values: List[float], quantiles: List[float]) -> List[float]:
    if not values:
        return []
    values = sorted(values)
    thresholds = []
    n = len(values)
    for q in quantiles:
        idx = int(round(q * (n - 1)))
        thresholds.append(values[idx])
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate between all-consensus and tc-only consensus.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Best-of-cycle JSON files")
    parser.add_argument(
        "--feature",
        choices=["tc_rate", "tc_count", "pairwise", "pairwise_tc", "consensus_margin"],
        default="tc_rate",
    )
    parser.add_argument("--direction", choices=["ge", "le"], default="ge")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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

    prepared = []
    for record in records:
        candidates = record.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            continue
        ground_truth = record.get("ground_truth", "")

        prepared_all: List[Tuple[dict, Counter[str]]] = []
        for entry in candidates:
            candidate = entry.get("candidate")
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            tokens = _counter(_tokenize(_normalize_statement(candidate)))
            prepared_all.append((entry, tokens))
        if not prepared_all:
            continue

        all_consensus, all_scores = _select_consensus(prepared_all)
        if all_consensus is None:
            continue

        prepared_tc = [(entry, tokens) for entry, tokens in prepared_all if entry.get("typecheck", False)]
        if prepared_tc:
            tc_consensus, _tc_scores = _select_consensus(prepared_tc)
        else:
            tc_consensus = all_consensus
        if tc_consensus is None:
            continue

        num_candidates = len(prepared_all)
        num_tc = len(prepared_tc)

        all_f1 = _statement_f1(str(all_consensus.get("candidate", "")), ground_truth)
        tc_f1 = _statement_f1(str(tc_consensus.get("candidate", "")), ground_truth)
        all_tc = bool(all_consensus.get("typecheck", False))
        tc_tc = bool(tc_consensus.get("typecheck", False))

        tokens_all = [tok for _, tok in prepared_all]
        tokens_tc = [tok for _, tok in prepared_tc]
        pairwise = _pairwise_avg(tokens_all)
        pairwise_tc = _pairwise_avg(tokens_tc) if len(tokens_tc) >= 2 else 0.0
        consensus_margin = 0.0
        if len(all_scores) > 1:
            sorted_scores = sorted(all_scores, reverse=True)
            consensus_margin = sorted_scores[0] - sorted_scores[1]

        prepared.append(
            {
                "all_f1": all_f1,
                "tc_f1": tc_f1,
                "all_tc": all_tc,
                "tc_tc": tc_tc,
                "tc_rate": num_tc / num_candidates,
                "tc_count": float(num_tc),
                "pairwise": pairwise,
                "pairwise_tc": pairwise_tc,
                "consensus_margin": consensus_margin,
            }
        )

    if not prepared:
        raise SystemExit("No usable records found.")

    feature_values = [p[args.feature] for p in prepared]
    thresholds = args.thresholds
    if thresholds is None:
        thresholds = _quantile_thresholds(feature_values, args.quantiles)
        thresholds = sorted(set(thresholds))

    summary = {}
    all_f1_avg = sum(p["all_f1"] for p in prepared) / len(prepared)
    tc_f1_avg = sum(p["tc_f1"] for p in prepared) / len(prepared)
    for threshold in thresholds:
        chosen_f1_sum = 0.0
        chosen_tc_sum = 0
        for p in prepared:
            select_tc = p[args.feature] >= threshold if args.direction == "ge" else p[args.feature] <= threshold
            if select_tc:
                chosen_f1_sum += p["tc_f1"]
                chosen_tc_sum += int(p["tc_tc"])
            else:
                chosen_f1_sum += p["all_f1"]
                chosen_tc_sum += int(p["all_tc"])
        summary[f"threshold_{threshold}"] = {
            "count": len(prepared),
            "chosen_f1_avg": chosen_f1_sum / len(prepared),
            "chosen_typecheck_rate": chosen_tc_sum / len(prepared),
        }

    if args.output is not None:
        payload = {
            "meta": {
                "results_paths": [str(p) for p in args.results],
                "feature": args.feature,
                "direction": args.direction,
                "dedupe_id": args.dedupe_id,
                "all_consensus_f1_avg": all_f1_avg,
                "tc_consensus_f1_avg": tc_f1_avg,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    print(f"  all_consensus_f1_avg: {all_f1_avg:.6f}")
    print(f"  tc_consensus_f1_avg: {tc_f1_avg:.6f}")
    best = max(summary.items(), key=lambda item: item[1]["chosen_f1_avg"])
    print(f"  best_threshold: {best[0]}")
    print(f"  best_chosen_f1_avg: {best[1]['chosen_f1_avg']:.6f}")


if __name__ == "__main__":
    main()
