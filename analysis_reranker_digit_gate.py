#!/usr/bin/env python3
"""Gate between base and NL-number learned rerankers using digit presence."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis_learned_reranker import _fit_ridge, _normalize_statement, _prepare_record


DIGIT_RE = re.compile(r"\d")
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


@dataclass
class ModelConfig:
    use_cycle: bool
    use_length: bool
    use_consensus: bool
    use_typecheck: bool
    extra_candidate_features: bool
    numeric_features: bool
    nl_number_features: bool
    nl_number_detail: bool
    nl_number_missing_extra: bool
    nl_number_missing_extra_ratio: bool
    nl_number_missing_extra_signed: bool


def _has_digits(record: dict, source: str) -> bool:
    if source == "nl":
        text = record.get("nl_statement", "") or ""
        return bool(DIGIT_RE.search(text) or WORD_NUM_RE.search(text))
    if source == "ground_truth":
        text = record.get("ground_truth", "") or ""
        stmt = _normalize_statement(text)
        return bool(DIGIT_RE.search(stmt) or WORD_NUM_RE.search(stmt))
    raise ValueError(f"Unknown digit source: {source}")


def _prepare(
    record: dict,
    config: ModelConfig,
) -> Tuple[List[List[float]], List[float], List[bool], float]:
    return _prepare_record(
        record,
        include_baseline=False,
        include_forward=False,
        include_cycle=config.use_cycle,
        include_length=config.use_length,
        include_consensus=config.use_consensus,
        include_typecheck=config.use_typecheck,
        include_embed_consensus=False,
        include_context_interactions=False,
        include_extra_candidate_features=config.extra_candidate_features,
        include_numeric_features=config.numeric_features,
        include_nl_number_features=config.nl_number_features,
        include_nl_number_detail=config.nl_number_detail,
        include_nl_number_missing_extra=config.nl_number_missing_extra,
        include_nl_number_missing_extra_ratio=config.nl_number_missing_extra_ratio,
        include_nl_number_missing_extra_signed=config.nl_number_missing_extra_signed,
        embedder=None,
    )


def _init_metrics() -> Dict[str, Dict[str, float]]:
    return {
        "count": 0,
        "f1_sum": 0.0,
        "baseline_sum": 0.0,
        "improve_sum": 0,
        "digits": {"count": 0, "f1_sum": 0.0, "baseline_sum": 0.0, "improve_sum": 0},
        "nodigits": {"count": 0, "f1_sum": 0.0, "baseline_sum": 0.0, "improve_sum": 0},
    }


def _update_metrics(metrics: Dict[str, Dict[str, float]], *, best_f1: float, baseline_f1: float, is_digit: bool) -> None:
    metrics["count"] += 1
    metrics["f1_sum"] += best_f1
    metrics["baseline_sum"] += baseline_f1
    metrics["improve_sum"] += int(best_f1 > baseline_f1)
    group = metrics["digits"] if is_digit else metrics["nodigits"]
    group["count"] += 1
    group["f1_sum"] += best_f1
    group["baseline_sum"] += baseline_f1
    group["improve_sum"] += int(best_f1 > baseline_f1)


def _finalize(metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    def _avg(total: float, count: int) -> float:
        return total / count if count else 0.0

    summary = {
        "count": metrics["count"],
        "f1_avg": _avg(metrics["f1_sum"], metrics["count"]),
        "baseline_f1_avg": _avg(metrics["baseline_sum"], metrics["count"]),
        "improve_rate": _avg(metrics["improve_sum"], metrics["count"]),
    }
    for label in ("digits", "nodigits"):
        group = metrics[label]
        summary[label] = {
            "count": group["count"],
            "f1_avg": _avg(group["f1_sum"], group["count"]),
            "baseline_f1_avg": _avg(group["baseline_sum"], group["count"]),
            "improve_rate": _avg(group["improve_sum"], group["count"]),
        }
    return summary


def _score_record(features: List[List[float]], targets: List[float], model: np.ndarray) -> float:
    scores = np.asarray(features, dtype=np.float64) @ model
    best_idx = int(np.argmax(scores))
    return targets[best_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate between rerankers based on digit presence.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--digit-source", choices=["nl", "ground_truth"], default="nl")
    parser.add_argument("--no-cycle", action="store_true", help="Drop cycle-score feature")
    parser.add_argument("--no-length", action="store_true", help="Drop length feature")
    parser.add_argument("--no-consensus", action="store_true", help="Drop consensus feature")
    parser.add_argument("--no-typecheck", action="store_true", help="Drop typecheck feature")
    parser.add_argument("--extra-candidate-features", action="store_true", help="Add per-candidate stats features")
    parser.add_argument("--numeric-features", action="store_true", help="Add numeric literal features")
    parser.add_argument("--nl-number-detail", action="store_true", help="Add detailed NL-number features")
    parser.add_argument("--nl-number-missing-extra", action="store_true", help="Add NL-number missing/extra features")
    parser.add_argument(
        "--nl-number-missing-extra-ratio",
        action="store_true",
        help="Add NL-number missing/extra ratio features",
    )
    parser.add_argument(
        "--nl-number-missing-extra-signed",
        action="store_true",
        help="Add NL-number signed delta feature",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base_config = ModelConfig(
        use_cycle=not args.no_cycle,
        use_length=not args.no_length,
        use_consensus=not args.no_consensus,
        use_typecheck=not args.no_typecheck,
        extra_candidate_features=args.extra_candidate_features,
        numeric_features=args.numeric_features,
        nl_number_features=False,
        nl_number_detail=False,
        nl_number_missing_extra=False,
        nl_number_missing_extra_ratio=False,
        nl_number_missing_extra_signed=False,
    )
    nl_config = ModelConfig(
        use_cycle=not args.no_cycle,
        use_length=not args.no_length,
        use_consensus=not args.no_consensus,
        use_typecheck=not args.no_typecheck,
        extra_candidate_features=args.extra_candidate_features,
        numeric_features=args.numeric_features,
        nl_number_features=True,
        nl_number_detail=args.nl_number_detail,
        nl_number_missing_extra=args.nl_number_missing_extra,
        nl_number_missing_extra_ratio=args.nl_number_missing_extra_ratio,
        nl_number_missing_extra_signed=args.nl_number_missing_extra_signed,
    )

    data = json.loads(args.results.read_text())
    records = data.get("records", [])

    prepared = []
    for record in records:
        base_features, base_targets, _base_tc, baseline_f1 = _prepare(record, base_config)
        nl_features, nl_targets, _nl_tc, _baseline_f1 = _prepare(record, nl_config)
        if not base_features or not nl_features:
            continue
        has_digits = _has_digits(record, args.digit_source)
        prepared.append(
            (base_features, base_targets, nl_features, nl_targets, baseline_f1, has_digits)
        )

    if not prepared:
        raise SystemExit("No usable records found.")

    k = max(2, args.folds)
    gate_metrics = _init_metrics()
    base_metrics = _init_metrics()
    nl_metrics = _init_metrics()
    gate_counts = {"use_nl_count": 0, "use_base_count": 0}

    for fold in range(k):
        X_train_base: List[List[float]] = []
        y_train_base: List[float] = []
        X_train_nl: List[List[float]] = []
        y_train_nl: List[float] = []
        test_records = []
        for idx, item in enumerate(prepared):
            if idx % k == fold:
                test_records.append(item)
            else:
                base_features, base_targets, nl_features, nl_targets, _baseline_f1, digit_flag = item
                X_train_base.extend(base_features)
                y_train_base.extend(base_targets)
                X_train_nl.extend(nl_features)
                y_train_nl.extend(nl_targets)

        model_base = _fit_ridge(
            np.asarray(X_train_base, dtype=np.float64),
            np.asarray(y_train_base, dtype=np.float64),
            args.ridge,
        )
        model_nl = _fit_ridge(
            np.asarray(X_train_nl, dtype=np.float64),
            np.asarray(y_train_nl, dtype=np.float64),
            args.ridge,
        )

        for base_features, base_targets, nl_features, nl_targets, baseline_f1, has_digits in test_records:
            base_f1 = _score_record(base_features, base_targets, model_base)
            nl_f1 = _score_record(nl_features, nl_targets, model_nl)
            _update_metrics(base_metrics, best_f1=base_f1, baseline_f1=baseline_f1, is_digit=has_digits)
            _update_metrics(nl_metrics, best_f1=nl_f1, baseline_f1=baseline_f1, is_digit=has_digits)

            if has_digits:
                gate_f1 = nl_f1
                gate_counts["use_nl_count"] += 1
            else:
                gate_f1 = base_f1
                gate_counts["use_base_count"] += 1
            _update_metrics(gate_metrics, best_f1=gate_f1, baseline_f1=baseline_f1, is_digit=has_digits)

    summary = {
        "base": _finalize(base_metrics),
        "nl": _finalize(nl_metrics),
        "gate": _finalize(gate_metrics),
        "gate_counts": gate_counts,
    }

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results),
                "folds": args.folds,
                "ridge": args.ridge,
                "digit_source": args.digit_source,
                "no_cycle": args.no_cycle,
                "no_length": args.no_length,
                "no_consensus": args.no_consensus,
                "no_typecheck": args.no_typecheck,
                "extra_candidate_features": args.extra_candidate_features,
                "numeric_features": args.numeric_features,
                "nl_number_detail": args.nl_number_detail,
                "nl_number_missing_extra": args.nl_number_missing_extra,
                "nl_number_missing_extra_ratio": args.nl_number_missing_extra_ratio,
                "nl_number_missing_extra_signed": args.nl_number_missing_extra_signed,
            },
            "summary": summary,
        }
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, block in summary.items():
        if key == "gate_counts":
            print(f"  gate_counts: {block}")
            continue
        print(f"  {key}: f1={block['f1_avg']:.6f} baseline={block['baseline_f1_avg']:.6f}")


if __name__ == "__main__":
    main()
