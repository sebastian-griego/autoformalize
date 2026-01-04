#!/usr/bin/env python3
"""Sweep combined cycle+consensus reranking with optional length penalty."""

from __future__ import annotations

import argparse
import json
import math
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


def _prepare_candidates(
    candidates: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    for entry in candidates:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        cycle = entry.get("cycle_score")
        if not isinstance(cycle, (int, float)):
            cycle = float("-inf")
        tokens = _counter(_tokenize(_normalize_statement(candidate)))
        prepared.append(
            {
                "entry": entry,
                "cycle": float(cycle),
                "tokens": tokens,
                "length": sum(tokens.values()),
                "consensus": 0.0,
            }
        )

    if len(prepared) <= 1:
        return prepared

    for idx, item in enumerate(prepared):
        score = 0.0
        for jdx, other in enumerate(prepared):
            if idx == jdx:
                continue
            score += _f1_score(item["tokens"], other["tokens"])
        item["consensus"] = score / max(1, len(prepared) - 1)
    return prepared


def _pick_candidate(
    prepared: List[Dict[str, object]],
    *,
    alpha: float,
    beta: float,
    require_typecheck: bool,
) -> Dict[str, object] | None:
    best = None
    best_score = float("-inf")
    for item in prepared:
        entry = item["entry"]
        if require_typecheck and not entry.get("typecheck", False):
            continue
        cycle = item["cycle"]
        if not math.isfinite(cycle):
            continue
        score = cycle - alpha * float(item["length"]) + beta * float(item["consensus"])
        if score > best_score:
            best_score = score
            best = entry
        elif score == best_score and best is not None:
            best_cycle = best.get("cycle_score")
            if isinstance(best_cycle, (int, float)) and math.isfinite(best_cycle) and cycle > float(best_cycle):
                best = entry
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep combined cycle+consensus reranking.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.001, 0.002, 0.005, 0.01, 0.02])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    summary: Dict[str, Dict[str, float]] = {}
    for alpha in args.alphas:
        for beta in args.betas:
            metrics = {
                "count": 0,
                "all_f1_sum": 0.0,
                "all_typecheck_sum": 0.0,
                "tc_f1_sum": 0.0,
                "tc_typecheck_sum": 0.0,
            }
            for record in records:
                candidates = record.get("candidates", [])
                if not isinstance(candidates, list) or not candidates:
                    continue
                prepared = _prepare_candidates(candidates)
                if not prepared:
                    continue
                ground_truth = record.get("ground_truth", "")
                pick_all = _pick_candidate(prepared, alpha=alpha, beta=beta, require_typecheck=False)
                pick_tc = _pick_candidate(prepared, alpha=alpha, beta=beta, require_typecheck=True)
                if pick_all is None:
                    baseline_candidate = record.get("baseline_candidate", "")
                    baseline_typecheck = bool(record.get("baseline_typecheck", False))
                    pick_all = {"candidate": baseline_candidate, "typecheck": baseline_typecheck}
                    pick_tc = pick_all
                if pick_tc is None:
                    pick_tc = pick_all

                cand_all = pick_all.get("candidate", "")
                cand_tc = pick_tc.get("candidate", "")
                metrics["count"] += 1
                metrics["all_f1_sum"] += _statement_f1(str(cand_all), ground_truth)
                metrics["all_typecheck_sum"] += int(bool(pick_all.get("typecheck", False)))
                metrics["tc_f1_sum"] += _statement_f1(str(cand_tc), ground_truth)
                metrics["tc_typecheck_sum"] += int(bool(pick_tc.get("typecheck", False)))

            key = f"alpha_{alpha}_beta_{beta}"
            if metrics["count"] == 0:
                summary[key] = {
                    "count": 0,
                    "all_f1_avg": 0.0,
                    "all_typecheck_rate": 0.0,
                    "tc_f1_avg": 0.0,
                    "tc_typecheck_rate": 0.0,
                }
            else:
                summary[key] = {
                    "count": metrics["count"],
                    "all_f1_avg": metrics["all_f1_sum"] / metrics["count"],
                    "all_typecheck_rate": metrics["all_typecheck_sum"] / metrics["count"],
                    "tc_f1_avg": metrics["tc_f1_sum"] / metrics["count"],
                    "tc_typecheck_rate": metrics["tc_typecheck_sum"] / metrics["count"],
                }

    if args.output is not None:
        payload = {"meta": {"results_path": str(args.results)}, "summary": summary}
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary (top 10 all-F1):")
    ranked = sorted(
        summary.items(),
        key=lambda item: item[1]["all_f1_avg"],
        reverse=True,
    )
    for key, metrics in ranked[:10]:
        print(
            f"{key} all_f1={metrics['all_f1_avg']:.3f} "
            f"all_tc={metrics['all_typecheck_rate']:.3f} "
            f"tc_f1={metrics['tc_f1_avg']:.3f} "
            f"tc_tc={metrics['tc_typecheck_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
