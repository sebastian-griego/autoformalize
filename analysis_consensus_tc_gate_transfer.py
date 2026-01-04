#!/usr/bin/env python3
"""Train a tc gate on one split and evaluate on another."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis_consensus_tc_gate_learned import (
    _fit_model,
    _predict_scores,
    _prepare_records,
    _zscore,
)


def _load_records(paths: list[Path], *, extra_stats: bool) -> list[dict]:
    records = []
    for path in paths:
        data = json.loads(path.read_text())
        records.extend([r for r in data.get("records", []) if isinstance(r, dict)])
    return _prepare_records(records, extra_stats=extra_stats)


def _select_threshold(train_preds: np.ndarray, train_records: list[dict], *, tune: bool) -> float:
    if not tune or not len(train_preds):
        return 0.0
    candidates = np.quantile(train_preds, np.linspace(0.0, 1.0, 21))
    best_thr = 0.0
    best_f1 = float("-inf")
    for thr in candidates:
        f1_sum = 0.0
        for pred, rec in zip(train_preds, train_records):
            choose_tc = pred > thr
            f1_sum += rec["tc_f1"] if choose_tc else rec["all_f1"]
        avg = f1_sum / len(train_records)
        if avg > best_f1:
            best_f1 = avg
            best_thr = float(thr)
    return best_thr


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tc gate on one split and test on another.")
    parser.add_argument("--train-results", type=Path, nargs="+", required=True)
    parser.add_argument("--test-results", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=["ridge", "gbrt"], default="ridge")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gbrt-estimators", type=int, default=200)
    parser.add_argument("--gbrt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbrt-max-depth", type=int, default=3)
    parser.add_argument("--extra-stats", action="store_true")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    train_records = _load_records(args.train_results, extra_stats=args.extra_stats)
    test_records = _load_records(args.test_results, extra_stats=args.extra_stats)

    if not train_records or not test_records:
        raise SystemExit("No usable records found for train/test.")

    X_train = np.asarray([r["features"] for r in train_records], dtype=np.float64)
    y_train = np.asarray([r["delta"] for r in train_records], dtype=np.float64)
    X_test = np.asarray([r["features"] for r in test_records], dtype=np.float64)

    X_train_z, mean, std = _zscore(X_train)
    X_test_z = (X_test - mean) / std

    model = _fit_model(
        X_train_z,
        y_train,
        model=args.model,
        ridge=args.ridge,
        gbrt_params={
            "n_estimators": args.gbrt_estimators,
            "learning_rate": args.gbrt_learning_rate,
            "max_depth": args.gbrt_max_depth,
        },
    )

    train_preds = _predict_scores(model, X_train_z, model_type=args.model)
    threshold = _select_threshold(train_preds, train_records, tune=args.tune_threshold)

    preds = _predict_scores(model, X_test_z, model_type=args.model)
    metrics = {
        "count": 0,
        "gate_f1_sum": 0.0,
        "all_f1_sum": 0.0,
        "tc_f1_sum": 0.0,
        "gate_typecheck_sum": 0.0,
        "choose_tc_sum": 0,
    }
    for pred, rec in zip(preds, test_records):
        choose_tc = pred > threshold
        gate_f1 = rec["tc_f1"] if choose_tc else rec["all_f1"]
        gate_tc = rec["tc_tc"] if choose_tc else rec["all_tc"]

        metrics["count"] += 1
        metrics["gate_f1_sum"] += gate_f1
        metrics["all_f1_sum"] += rec["all_f1"]
        metrics["tc_f1_sum"] += rec["tc_f1"]
        metrics["gate_typecheck_sum"] += int(bool(gate_tc))
        metrics["choose_tc_sum"] += int(bool(choose_tc))

    summary = {
        "count": metrics["count"],
        "gate_f1_avg": metrics["gate_f1_sum"] / metrics["count"],
        "all_f1_avg": metrics["all_f1_sum"] / metrics["count"],
        "tc_f1_avg": metrics["tc_f1_sum"] / metrics["count"],
        "gate_typecheck_rate": metrics["gate_typecheck_sum"] / metrics["count"],
        "choose_tc_rate": metrics["choose_tc_sum"] / metrics["count"],
        "choose_all_rate": 1.0 - (metrics["choose_tc_sum"] / metrics["count"]),
        "threshold": threshold,
    }

    if args.model == "ridge":
        summary["weights"] = model.tolist()
    else:
        summary["importances"] = model.feature_importances_.tolist()

    if args.output is not None:
        payload = {
            "meta": {
                "train_paths": [str(p) for p in args.train_results],
                "test_paths": [str(p) for p in args.test_results],
                "model": args.model,
                "ridge": args.ridge,
                "gbrt_params": {
                    "n_estimators": args.gbrt_estimators,
                    "learning_rate": args.gbrt_learning_rate,
                    "max_depth": args.gbrt_max_depth,
                },
                "extra_stats": args.extra_stats,
                "tune_threshold": args.tune_threshold,
            },
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
