#!/usr/bin/env python3
"""Leave-one-slice-out evaluation for the learned reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_learned_reranker import _fit_ridge, _prepare_record


def _load_slice(
    path: Path,
    *,
    baseline_feature: bool,
    include_cycle: bool,
    include_length: bool,
    include_consensus: bool,
    include_typecheck: bool,
    include_context_interactions: bool,
    include_extra_candidate_features: bool,
    include_numeric_features: bool,
    include_nl_number_features: bool,
    include_nl_number_detail: bool,
    include_nl_number_missing_extra: bool,
    include_nl_number_missing_extra_ratio: bool,
    include_nl_number_missing_extra_signed: bool,
) -> list:
    data = json.loads(path.read_text())
    prepared = []
    for record in data.get("records", []):
        features, targets, typechecks, baseline_f1 = _prepare_record(
            record,
            include_baseline=baseline_feature,
            include_forward=False,
            include_cycle=include_cycle,
            include_length=include_length,
            include_consensus=include_consensus,
            include_typecheck=include_typecheck,
            include_embed_consensus=False,
            include_context_interactions=include_context_interactions,
            include_extra_candidate_features=include_extra_candidate_features,
            include_numeric_features=include_numeric_features,
            include_nl_number_features=include_nl_number_features,
            include_nl_number_detail=include_nl_number_detail,
            include_nl_number_missing_extra=include_nl_number_missing_extra,
            include_nl_number_missing_extra_ratio=include_nl_number_missing_extra_ratio,
            include_nl_number_missing_extra_signed=include_nl_number_missing_extra_signed,
            embedder=None,
        )
        if features:
            prepared.append((features, targets, typechecks, baseline_f1))
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-slice-out CV for the learned reranker.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Slice JSON files")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--baseline-feature", action="store_true", help="Include baseline similarity feature")
    parser.add_argument("--no-cycle", action="store_true", help="Drop cycle-score feature")
    parser.add_argument("--no-length", action="store_true", help="Drop length feature")
    parser.add_argument("--no-consensus", action="store_true", help="Drop consensus feature")
    parser.add_argument("--no-typecheck", action="store_true", help="Drop typecheck feature")
    parser.add_argument("--context-interactions", action="store_true", help="Add consensus/tc interaction features")
    parser.add_argument("--extra-candidate-features", action="store_true", help="Add per-candidate stats features")
    parser.add_argument("--numeric-features", action="store_true", help="Add numeric literal features")
    parser.add_argument("--nl-number-features", action="store_true", help="Add NL-number overlap features")
    parser.add_argument("--nl-number-detail", action="store_true", help="Add detailed NL-number features")
    parser.add_argument(
        "--nl-number-missing-extra",
        action="store_true",
        help="Add NL-number missing/extra features",
    )
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

    slices = {}
    for path in args.results:
        prepared = _load_slice(
            path,
            baseline_feature=args.baseline_feature,
            include_cycle=not args.no_cycle,
            include_length=not args.no_length,
            include_consensus=not args.no_consensus,
            include_typecheck=not args.no_typecheck,
            include_context_interactions=args.context_interactions,
            include_extra_candidate_features=args.extra_candidate_features,
            include_numeric_features=args.numeric_features,
            include_nl_number_features=args.nl_number_features or args.nl_number_detail,
            include_nl_number_detail=args.nl_number_detail,
            include_nl_number_missing_extra=args.nl_number_missing_extra,
            include_nl_number_missing_extra_ratio=args.nl_number_missing_extra_ratio,
            include_nl_number_missing_extra_signed=args.nl_number_missing_extra_signed,
        )
        if prepared:
            slices[path.name] = prepared

    if not slices:
        raise SystemExit("No usable records found.")

    per_slice = {}
    total = {
        "count": 0,
        "baseline_f1_sum": 0.0,
        "learned_f1_sum": 0.0,
        "learned_tc_f1_sum": 0.0,
        "learned_typecheck_sum": 0.0,
        "learned_tc_typecheck_sum": 0.0,
    }

    for test_name, test_records in slices.items():
        train_records = []
        for name, records in slices.items():
            if name != test_name:
                train_records.extend(records)

        X_train = []
        y_train = []
        for features, targets, _typechecks, _baseline_f1 in train_records:
            X_train.extend(features)
            y_train.extend(targets)

        X_train_np = np.asarray(X_train, dtype=np.float64)
        y_train_np = np.asarray(y_train, dtype=np.float64)
        model = _fit_ridge(X_train_np, y_train_np, args.ridge)

        metrics = {
            "count": 0,
            "baseline_f1_sum": 0.0,
            "learned_f1_sum": 0.0,
            "learned_tc_f1_sum": 0.0,
            "learned_typecheck_sum": 0.0,
            "learned_tc_typecheck_sum": 0.0,
        }
        for features, targets, typechecks, baseline_f1 in test_records:
            scores = np.asarray(features, dtype=np.float64) @ model
            best_idx = int(np.argmax(scores))
            best_f1 = targets[best_idx]
            best_tc = typechecks[best_idx]

            tc_indices = [i for i, tc in enumerate(typechecks) if tc]
            if tc_indices:
                tc_scores = [scores[i] for i in tc_indices]
                best_tc_idx = tc_indices[int(np.argmax(tc_scores))]
            else:
                best_tc_idx = best_idx
            best_tc_f1 = targets[best_tc_idx]
            best_tc_tc = typechecks[best_tc_idx]

            metrics["count"] += 1
            metrics["baseline_f1_sum"] += baseline_f1
            metrics["learned_f1_sum"] += best_f1
            metrics["learned_tc_f1_sum"] += best_tc_f1
            metrics["learned_typecheck_sum"] += int(best_tc)
            metrics["learned_tc_typecheck_sum"] += int(best_tc_tc)

        per_slice[test_name] = {
            "count": metrics["count"],
            "baseline_f1_avg": metrics["baseline_f1_sum"] / metrics["count"],
            "learned_f1_avg": metrics["learned_f1_sum"] / metrics["count"],
            "learned_tc_f1_avg": metrics["learned_tc_f1_sum"] / metrics["count"],
            "learned_typecheck_rate": metrics["learned_typecheck_sum"] / metrics["count"],
            "learned_tc_typecheck_rate": metrics["learned_tc_typecheck_sum"] / metrics["count"],
        }

        total["count"] += metrics["count"]
        total["baseline_f1_sum"] += metrics["baseline_f1_sum"]
        total["learned_f1_sum"] += metrics["learned_f1_sum"]
        total["learned_tc_f1_sum"] += metrics["learned_tc_f1_sum"]
        total["learned_typecheck_sum"] += metrics["learned_typecheck_sum"]
        total["learned_tc_typecheck_sum"] += metrics["learned_tc_typecheck_sum"]

    summary = {
        "count": total["count"],
        "baseline_f1_avg": total["baseline_f1_sum"] / total["count"],
        "learned_f1_avg": total["learned_f1_sum"] / total["count"],
        "learned_tc_f1_avg": total["learned_tc_f1_sum"] / total["count"],
        "learned_typecheck_rate": total["learned_typecheck_sum"] / total["count"],
        "learned_tc_typecheck_rate": total["learned_tc_typecheck_sum"] / total["count"],
    }

    if args.output is not None:
        payload = {
            "meta": {
                "results_paths": [str(p) for p in args.results],
                "ridge": args.ridge,
                "baseline_feature": args.baseline_feature,
                "no_cycle": args.no_cycle,
                "no_length": args.no_length,
                "no_consensus": args.no_consensus,
                "no_typecheck": args.no_typecheck,
                "context_interactions": args.context_interactions,
                "extra_candidate_features": args.extra_candidate_features,
                "numeric_features": args.numeric_features,
                "nl_number_features": args.nl_number_features or args.nl_number_detail,
                "nl_number_detail": args.nl_number_detail,
                "nl_number_missing_extra": args.nl_number_missing_extra,
                "nl_number_missing_extra_ratio": args.nl_number_missing_extra_ratio,
                "nl_number_missing_extra_signed": args.nl_number_missing_extra_signed,
            },
            "summary": summary,
            "per_slice": per_slice,
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
