#!/usr/bin/env python3
"""Learn a gate between all-consensus and tc-only consensus using record-level features."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def _fit_model(X: np.ndarray, y: np.ndarray, *, model: str, ridge: float, gbrt_params: dict) -> object:
    if model == "ridge":
        return _fit_ridge(X, y, ridge)
    if model == "gbrt":
        from sklearn.ensemble import GradientBoostingRegressor

        reg = GradientBoostingRegressor(
            n_estimators=gbrt_params["n_estimators"],
            learning_rate=gbrt_params["learning_rate"],
            max_depth=gbrt_params["max_depth"],
            random_state=0,
        )
        reg.fit(X, y)
        return reg
    raise ValueError(f"Unknown model: {model}")


def _predict_scores(model: object, X: np.ndarray, *, model_type: str) -> np.ndarray:
    if model_type == "ridge":
        return X @ model
    if model_type == "gbrt":
        return model.predict(X)
    raise ValueError(f"Unknown model: {model_type}")


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


def _zscore(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (values - mean) / std, mean, std


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = X.T @ X
    xtx += ridge * np.eye(X.shape[1])
    return np.linalg.solve(xtx, X.T @ y)


def _stat_max_mean(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return max(values), sum(values) / len(values)


def _prepare_records(records: List[dict], *, extra_stats: bool) -> List[dict]:
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
            tc_consensus, _ = _select_consensus(prepared_tc)
        else:
            tc_consensus = all_consensus
        if tc_consensus is None:
            continue

        num_candidates = len(prepared_all)
        num_tc = len(prepared_tc)
        tc_rate = num_tc / num_candidates if num_candidates else 0.0

        tokens_all = [tok for _, tok in prepared_all]
        tokens_tc = [tok for _, tok in prepared_tc]
        pairwise = _pairwise_avg(tokens_all)
        pairwise_tc = _pairwise_avg(tokens_tc) if len(tokens_tc) >= 2 else 0.0
        consensus_margin = 0.0
        if len(all_scores) > 1:
            sorted_scores = sorted(all_scores, reverse=True)
            consensus_margin = sorted_scores[0] - sorted_scores[1]

        extra_features: List[float] = []
        if extra_stats:
            cycle_scores = []
            lenpen_scores = []
            lengths = []
            best_cycle_score = None
            best_cycle_tc = 0.0
            best_lenpen_score = None
            best_lenpen_tc = 0.0
            for entry, tok in prepared_all:
                cycle = entry.get("cycle_score")
                if isinstance(cycle, (int, float)) and math.isfinite(cycle):
                    cycle_scores.append(float(cycle))
                    if best_cycle_score is None or cycle > best_cycle_score:
                        best_cycle_score = float(cycle)
                        best_cycle_tc = 1.0 if entry.get("typecheck", False) else 0.0
                lenpen = entry.get("length_penalized_score")
                if isinstance(lenpen, (int, float)) and math.isfinite(lenpen):
                    lenpen_scores.append(float(lenpen))
                    if best_lenpen_score is None or lenpen > best_lenpen_score:
                        best_lenpen_score = float(lenpen)
                        best_lenpen_tc = 1.0 if entry.get("typecheck", False) else 0.0
                stmt_len = entry.get("statement_len")
                if isinstance(stmt_len, (int, float)) and math.isfinite(stmt_len):
                    lengths.append(float(stmt_len))
                else:
                    lengths.append(float(len(tok)))

            tc_cycle_scores = []
            tc_lenpen_scores = []
            tc_lengths = []
            for entry, tok in prepared_tc:
                cycle = entry.get("cycle_score")
                if isinstance(cycle, (int, float)) and math.isfinite(cycle):
                    tc_cycle_scores.append(float(cycle))
                lenpen = entry.get("length_penalized_score")
                if isinstance(lenpen, (int, float)) and math.isfinite(lenpen):
                    tc_lenpen_scores.append(float(lenpen))
                stmt_len = entry.get("statement_len")
                if isinstance(stmt_len, (int, float)) and math.isfinite(stmt_len):
                    tc_lengths.append(float(stmt_len))
                else:
                    tc_lengths.append(float(len(tok)))

            cycle_max, cycle_mean = _stat_max_mean(cycle_scores)
            tc_cycle_max, tc_cycle_mean = _stat_max_mean(tc_cycle_scores)
            lenpen_max, lenpen_mean = _stat_max_mean(lenpen_scores)
            tc_lenpen_max, tc_lenpen_mean = _stat_max_mean(tc_lenpen_scores)
            len_mean = sum(lengths) / len(lengths) if lengths else 0.0
            tc_len_mean = sum(tc_lengths) / len(tc_lengths) if tc_lengths else 0.0

            extra_features = [
                cycle_max,
                cycle_mean,
                tc_cycle_max,
                tc_cycle_mean,
                lenpen_max,
                lenpen_mean,
                tc_lenpen_max,
                tc_lenpen_mean,
                len_mean,
                tc_len_mean,
                best_cycle_tc,
                best_lenpen_tc,
            ]

        all_f1 = _statement_f1(str(all_consensus.get("candidate", "")), ground_truth)
        tc_f1 = _statement_f1(str(tc_consensus.get("candidate", "")), ground_truth)
        delta = tc_f1 - all_f1

        prepared.append(
            {
                "features": [
                    1.0,
                    float(num_candidates),
                    float(num_tc),
                    tc_rate,
                    pairwise,
                    pairwise_tc,
                    consensus_margin,
                ]
                + extra_features,
                "delta": delta,
                "all_f1": all_f1,
                "tc_f1": tc_f1,
                "all_tc": bool(all_consensus.get("typecheck", False)),
                "tc_tc": bool(tc_consensus.get("typecheck", False)),
            }
        )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a gate between all-consensus and tc-only consensus.")
    parser.add_argument("--results", type=Path, nargs="+", required=True, help="Best-of-cycle JSON files")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", choices=["ridge", "gbrt"], default="ridge")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gbrt-estimators", type=int, default=200)
    parser.add_argument("--gbrt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbrt-max-depth", type=int, default=3)
    parser.add_argument("--extra-stats", action="store_true", help="Include cycle/lenpen/length summary stats")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune tc selection threshold on train folds")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = []
    for path in args.results:
        data = json.loads(path.read_text())
        records.extend([r for r in data.get("records", []) if isinstance(r, dict)])

    prepared = _prepare_records(records, extra_stats=args.extra_stats)
    if not prepared:
        raise SystemExit("No usable records found.")

    k = max(2, args.folds)
    fold_metrics = []
    weights = []
    thresholds = []

    for fold in range(k):
        train_rows = []
        train_targets = []
        train_records = []
        test_rows = []
        test_records = []
        for idx, rec in enumerate(prepared):
            if idx % k == fold:
                test_rows.append(rec["features"])
                test_records.append(rec)
            else:
                train_rows.append(rec["features"])
                train_targets.append(rec["delta"])
                train_records.append(rec)

        X_train = np.asarray(train_rows, dtype=np.float64)
        y_train = np.asarray(train_targets, dtype=np.float64)
        X_test = np.asarray(test_rows, dtype=np.float64)

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

        metrics = {
            "count": 0,
            "gate_f1_sum": 0.0,
            "all_f1_sum": 0.0,
            "tc_f1_sum": 0.0,
            "gate_typecheck_sum": 0.0,
        }
        threshold = 0.0
        if args.tune_threshold:
            train_preds = _predict_scores(model, X_train_z, model_type=args.model)
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
            threshold = best_thr
        thresholds.append(threshold)

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

        fold_metrics.append(metrics)

    total_count = sum(m["count"] for m in fold_metrics)
    summary = {
        "count": total_count,
        "gate_f1_avg": sum(m["gate_f1_sum"] for m in fold_metrics) / total_count,
        "all_f1_avg": sum(m["all_f1_sum"] for m in fold_metrics) / total_count,
        "tc_f1_avg": sum(m["tc_f1_sum"] for m in fold_metrics) / total_count,
        "gate_typecheck_rate": sum(m["gate_typecheck_sum"] for m in fold_metrics) / total_count,
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
                "folds": args.folds,
                "model": args.model,
                "ridge": args.ridge,
                "gbrt_params": {
                    "n_estimators": args.gbrt_estimators,
                    "learning_rate": args.gbrt_learning_rate,
                    "max_depth": args.gbrt_max_depth,
                },
                "features": [
                    "bias",
                    "count",
                    "tc_count",
                    "tc_rate",
                    "pairwise",
                    "pairwise_tc",
                    "consensus_margin",
                ]
                + (
                    [
                        "cycle_max",
                        "cycle_mean",
                        "tc_cycle_max",
                        "tc_cycle_mean",
                        "lenpen_max",
                        "lenpen_mean",
                        "tc_lenpen_max",
                        "tc_lenpen_mean",
                        "len_mean",
                        "tc_len_mean",
                        "best_cycle_is_tc",
                        "best_lenpen_is_tc",
                    ]
                    if args.extra_stats
                    else []
                ),
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
