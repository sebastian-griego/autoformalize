#!/usr/bin/env python3
"""Sweep gating thresholds to choose baseline vs consensus."""

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


def _as_float(value: object) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) else None


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
    parser = argparse.ArgumentParser(description="Gate between baseline and consensus.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Best-of-cycle JSON files")
    parser.add_argument(
        "--feature",
        choices=[
            "count",
            "tc_rate",
            "pairwise",
            "pairwise_tc",
            "consensus_margin",
            "baseline_cycle",
            "baseline_len",
            "baseline_lenpen",
        ],
        default="pairwise",
    )
    parser.add_argument("--direction", choices=["le", "ge"], default="le")
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

    per_record = []
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
            tok = _counter(_tokenize(_normalize_statement(candidate)))
            tokens.append(tok)
            typechecks.append(bool(entry.get("typecheck", False)))
            f1s.append(_statement_f1(candidate, ground_truth))

        if not texts:
            continue

        baseline_f1 = _statement_f1(baseline_candidate, ground_truth)
        baseline_tc = baseline_typecheck
        consensus_scores = _consensus_scores(tokens)
        best_idx = max(range(len(texts)), key=lambda i: consensus_scores[i])
        consensus_f1 = f1s[best_idx]
        consensus_tc = typechecks[best_idx]

        count_val = float(len(texts))
        tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0
        pairwise = _pairwise_avg(tokens)
        tc_tokens = [tok for tok, tc in zip(tokens, typechecks) if tc]
        pairwise_tc = _pairwise_avg(tc_tokens) if len(tc_tokens) >= 2 else 0.0
        sorted_scores = sorted(consensus_scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        baseline_len = _as_float(record.get("baseline_statement_len"))
        if baseline_len is None and baseline_candidate:
            baseline_len = float(len(_tokenize(_normalize_statement(baseline_candidate))))

        per_record.append(
            {
                "baseline_f1": baseline_f1,
                "baseline_tc": baseline_tc,
                "consensus_f1": consensus_f1,
                "consensus_tc": consensus_tc,
                "count": count_val,
                "tc_rate": tc_rate,
                "pairwise": pairwise,
                "pairwise_tc": pairwise_tc,
                "consensus_margin": margin,
                "baseline_cycle": _as_float(record.get("baseline_cycle_score")),
                "baseline_len": baseline_len,
                "baseline_lenpen": _as_float(record.get("baseline_length_penalized_score")),
            }
        )

    if not per_record:
        raise SystemExit("No usable records.")

    values = [rec[args.feature] for rec in per_record if rec.get(args.feature) is not None]
    thresholds = args.thresholds or _quantile_thresholds(values, args.quantiles)

    summary = []
    for threshold in thresholds:
        f1_sum = 0.0
        tc_sum = 0.0
        count = 0
        for rec in per_record:
            feature_val = rec.get(args.feature)
            if feature_val is None:
                continue
            choose_consensus = feature_val <= threshold if args.direction == "le" else feature_val >= threshold
            if choose_consensus:
                f1_sum += rec["consensus_f1"]
                tc_sum += int(bool(rec["consensus_tc"]))
            else:
                f1_sum += rec["baseline_f1"]
                tc_sum += int(bool(rec["baseline_tc"]))
            count += 1
        summary.append(
            {
                "threshold": threshold,
                "f1_avg": f1_sum / count if count else 0.0,
                "typecheck_rate": tc_sum / count if count else 0.0,
            }
        )

    if args.output is not None:
        payload = {
            "meta": {
                "results_paths": [str(p) for p in args.results],
                "feature": args.feature,
                "direction": args.direction,
                "thresholds": thresholds,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for entry in summary:
        print(
            f"  thr={entry['threshold']:.4f} f1={entry['f1_avg']:.3f} tc={entry['typecheck_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
