#!/usr/bin/env python3
"""Gate between two learned rerankers based on candidate count."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from analysis_learned_reranker import _fit_ridge, _prepare_record


@dataclass
class ModelConfig:
    use_cycle: bool
    extra_candidate_features: bool


def _load_training(path: Path, config: ModelConfig) -> List[tuple]:
    data = json.loads(path.read_text())
    prepared = []
    for record in data.get("records", []):
        features, targets, typechecks, baseline_f1 = _prepare_record(
            record,
            include_baseline=False,
            include_forward=False,
            include_cycle=config.use_cycle,
            include_length=True,
            include_consensus=True,
            include_typecheck=True,
            include_embed_consensus=False,
            include_context_interactions=False,
            include_extra_candidate_features=config.extra_candidate_features,
            include_numeric_features=False,
            include_nl_number_features=False,
            embedder=None,
        )
        if features:
            prepared.append((features, targets, typechecks, baseline_f1))
    return prepared


def _fit_model(prepared: List[tuple], ridge: float) -> np.ndarray:
    X_train = []
    y_train = []
    for features, targets, _typechecks, _baseline_f1 in prepared:
        X_train.extend(features)
        y_train.extend(targets)
    X_train_np = np.asarray(X_train, dtype=np.float64)
    y_train_np = np.asarray(y_train, dtype=np.float64)
    return _fit_ridge(X_train_np, y_train_np, ridge)


def _score_record(record: dict, model: np.ndarray, config: ModelConfig) -> dict | None:
    features, targets, typechecks, baseline_f1 = _prepare_record(
        record,
        include_baseline=False,
        include_forward=False,
        include_cycle=config.use_cycle,
        include_length=True,
        include_consensus=True,
        include_typecheck=True,
        include_embed_consensus=False,
        include_context_interactions=False,
        include_extra_candidate_features=config.extra_candidate_features,
        include_numeric_features=False,
        include_nl_number_features=False,
        embedder=None,
    )
    if not features:
        return None
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

    candidates = record.get("candidates", [])
    cand_count = len(candidates) if isinstance(candidates, list) else 0

    return {
        "count": cand_count,
        "baseline_f1": baseline_f1,
        "best_f1": best_f1,
        "best_tc_f1": best_tc_f1,
        "best_tc": bool(best_tc),
        "best_tc_tc": bool(best_tc_tc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate between two rerankers by candidate count.")
    parser.add_argument("--low-results", type=Path, required=True, help="Training data for low-count model")
    parser.add_argument("--high-results", type=Path, required=True, help="Training data for high-count model")
    parser.add_argument("--test-results", type=Path, nargs="+", required=True, help="Test JSON files")
    parser.add_argument("--low-use-cycle", action="store_true", help="Enable cycle feature for low model")
    parser.add_argument("--high-use-cycle", action="store_true", help="Enable cycle feature for high model")
    parser.add_argument(
        "--low-extra-candidate-features",
        action="store_true",
        help="Enable extra candidate features for low model",
    )
    parser.add_argument(
        "--high-extra-candidate-features",
        action="store_true",
        help="Enable extra candidate features for high model",
    )
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 64, 80, 96, 112, 128, 160])
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    low_config = ModelConfig(
        use_cycle=args.low_use_cycle,
        extra_candidate_features=args.low_extra_candidate_features,
    )
    high_config = ModelConfig(
        use_cycle=args.high_use_cycle,
        extra_candidate_features=args.high_extra_candidate_features,
    )

    low_train = _load_training(args.low_results, low_config)
    high_train = _load_training(args.high_results, high_config)
    if not low_train or not high_train:
        raise SystemExit("Missing training data for low/high models.")

    low_model = _fit_model(low_train, args.ridge)
    high_model = _fit_model(high_train, args.ridge)

    test_records = []
    for path in args.test_results:
        data = json.loads(path.read_text())
        test_records.extend([r for r in data.get("records", []) if isinstance(r, dict)])

    scored = []
    for record in test_records:
        low = _score_record(record, low_model, low_config)
        high = _score_record(record, high_model, high_config)
        if low is None or high is None:
            continue
        scored.append(
            {
                "count": low["count"],
                "baseline_f1": low["baseline_f1"],
                "low": low,
                "high": high,
            }
        )

    if not scored:
        raise SystemExit("No usable test records.")

    summary = {
        "count": len(scored),
        "baseline_f1_avg": sum(r["baseline_f1"] for r in scored) / len(scored),
        "low_f1_avg": sum(r["low"]["best_f1"] for r in scored) / len(scored),
        "high_f1_avg": sum(r["high"]["best_f1"] for r in scored) / len(scored),
    }

    gates = {}
    for threshold in args.thresholds:
        metrics = {
            "count": 0,
            "gate_f1_sum": 0.0,
            "gate_tc_f1_sum": 0.0,
            "gate_typecheck_sum": 0.0,
            "choose_high_sum": 0,
        }
        for rec in scored:
            use_high = rec["count"] >= threshold
            pick = rec["high"] if use_high else rec["low"]
            metrics["count"] += 1
            metrics["gate_f1_sum"] += pick["best_f1"]
            metrics["gate_tc_f1_sum"] += pick["best_tc_f1"]
            metrics["gate_typecheck_sum"] += int(bool(pick["best_tc"]))
            metrics["choose_high_sum"] += int(use_high)
        gates[f"threshold_{threshold}"] = {
            "count": metrics["count"],
            "gate_f1_avg": metrics["gate_f1_sum"] / metrics["count"],
            "gate_tc_f1_avg": metrics["gate_tc_f1_sum"] / metrics["count"],
            "gate_typecheck_rate": metrics["gate_typecheck_sum"] / metrics["count"],
            "choose_high_rate": metrics["choose_high_sum"] / metrics["count"],
        }

    best = max(gates.items(), key=lambda item: item[1]["gate_f1_avg"])
    summary["best_threshold"] = best[0]
    summary["best_gate_f1_avg"] = best[1]["gate_f1_avg"]

    payload = {
        "meta": {
            "low_results": str(args.low_results),
            "high_results": str(args.high_results),
            "test_results": [str(p) for p in args.test_results],
            "low_config": low_config.__dict__,
            "high_config": high_config.__dict__,
            "thresholds": args.thresholds,
            "ridge": args.ridge,
        },
        "summary": summary,
        "gates": gates,
    }

    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
