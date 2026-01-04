#!/usr/bin/env python3
"""Leave-one-slice-out evaluation for the learned tc gate."""

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


def _load_slice(path: Path, *, extra_stats: bool) -> list[dict]:
    data = json.loads(path.read_text())
    records = [r for r in data.get("records", []) if isinstance(r, dict)]
    return _prepare_records(records, extra_stats=extra_stats)


def _select_threshold(
    train_preds: np.ndarray, train_records: list[dict], *, tune: bool
) -> float:
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
    parser = argparse.ArgumentParser(description="Leave-one-slice-out learned gate evaluation.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Slice JSON files")
    parser.add_argument("--model", choices=["ridge", "gbrt"], default="ridge")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gbrt-estimators", type=int, default=200)
    parser.add_argument("--gbrt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbrt-max-depth", type=int, default=3)
    parser.add_argument("--extra-stats", action="store_true", help="Include cycle/lenpen/length summary stats")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune tc selection threshold on train slices")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    slices: dict[str, list[dict]] = {}
    for path in args.results:
        prepared = _load_slice(path, extra_stats=args.extra_stats)
        if prepared:
            slices[path.name] = prepared

    if not slices:
        raise SystemExit("No usable records found.")

    per_slice = {}
    weights = []
    thresholds = []

    totals = {
        "count": 0,
        "gate_f1_sum": 0.0,
        "all_f1_sum": 0.0,
        "tc_f1_sum": 0.0,
        "gate_typecheck_sum": 0.0,
        "choose_tc_sum": 0,
    }

    for test_name, test_records in slices.items():
        train_records = []
        for name, records in slices.items():
            if name != test_name:
                train_records.extend(records)

        if not train_records or not test_records:
            continue

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
        if args.model == "ridge":
            weights.append(model.tolist())
        else:
            weights.append(model.feature_importances_.tolist())

        train_preds = _predict_scores(model, X_train_z, model_type=args.model)
        threshold = _select_threshold(train_preds, train_records, tune=args.tune_threshold)
        thresholds.append(threshold)

        metrics = {
            "count": 0,
            "gate_f1_sum": 0.0,
            "all_f1_sum": 0.0,
            "tc_f1_sum": 0.0,
            "gate_typecheck_sum": 0.0,
            "choose_tc_sum": 0,
        }

        preds = _predict_scores(model, X_test_z, model_type=args.model)
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

        per_slice[test_name] = {
            "count": metrics["count"],
            "gate_f1_avg": metrics["gate_f1_sum"] / metrics["count"],
            "all_f1_avg": metrics["all_f1_sum"] / metrics["count"],
            "tc_f1_avg": metrics["tc_f1_sum"] / metrics["count"],
            "gate_typecheck_rate": metrics["gate_typecheck_sum"] / metrics["count"],
            "choose_tc_rate": metrics["choose_tc_sum"] / metrics["count"],
        }

        totals["count"] += metrics["count"]
        totals["gate_f1_sum"] += metrics["gate_f1_sum"]
        totals["all_f1_sum"] += metrics["all_f1_sum"]
        totals["tc_f1_sum"] += metrics["tc_f1_sum"]
        totals["gate_typecheck_sum"] += metrics["gate_typecheck_sum"]
        totals["choose_tc_sum"] += metrics["choose_tc_sum"]

    if totals["count"] == 0:
        raise SystemExit("No usable records after slicing.")

    summary = {
        "count": totals["count"],
        "gate_f1_avg": totals["gate_f1_sum"] / totals["count"],
        "all_f1_avg": totals["all_f1_sum"] / totals["count"],
        "tc_f1_avg": totals["tc_f1_sum"] / totals["count"],
        "gate_typecheck_rate": totals["gate_typecheck_sum"] / totals["count"],
        "choose_tc_rate": totals["choose_tc_sum"] / totals["count"],
        "choose_all_rate": 1.0 - (totals["choose_tc_sum"] / totals["count"]),
        "gate_beats_all_slices": sum(
            1 for v in per_slice.values() if v["gate_f1_avg"] > v["all_f1_avg"]
        ),
        "gate_beats_tc_slices": sum(
            1 for v in per_slice.values() if v["gate_f1_avg"] > v["tc_f1_avg"]
        ),
        "slices": len(per_slice),
    }
    if args.model == "ridge":
        summary["avg_weights"] = np.mean(np.asarray(weights), axis=0).tolist()
    else:
        summary["avg_importances"] = np.mean(np.asarray(weights), axis=0).tolist()
    if args.tune_threshold:
        summary["avg_threshold"] = float(np.mean(np.asarray(thresholds)))

    if args.output is not None:
        payload = {
            "meta": {
                "results_paths": [str(p) for p in args.results],
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
