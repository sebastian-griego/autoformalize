#!/usr/bin/env python3
"""Learn a meta-selector between consensus and learned reranker outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


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


def _consensus_index(tokens: List[Counter[str]]) -> Tuple[int, List[float]]:
    if len(tokens) <= 1:
        return 0, [0.0 for _ in tokens]
    best_idx = 0
    best_score = float("-inf")
    scores: List[float] = []
    for i, tok in enumerate(tokens):
        score = 0.0
        for j, other in enumerate(tokens):
            if i == j:
                continue
            score += _f1_score(tok, other)
        score /= max(1, len(tokens) - 1)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, scores


def _zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    xtx = X.T @ X
    xtx += ridge * np.eye(X.shape[1])
    return np.linalg.solve(xtx, X.T @ y)


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


def _prepare_record(record: dict) -> dict | None:
    prepared = []
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return None
    ground_truth = record.get("ground_truth", "")
    if not isinstance(ground_truth, str):
        return None

    baseline_candidate = record.get("baseline_candidate", "")
    baseline_typecheck = bool(record.get("baseline_typecheck", False))

    entries = list(candidates)
    if baseline_candidate and not any(
        isinstance(entry, dict) and entry.get("candidate") == baseline_candidate
        for entry in entries
    ):
        entries.append({"candidate": baseline_candidate, "typecheck": baseline_typecheck})

    texts: List[str] = []
    typechecks: List[bool] = []
    cycles: List[float] = []
    lengths: List[float] = []
    tokens: List[Counter[str]] = []
    targets: List[float] = []
    for entry in entries:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        texts.append(candidate)
        tc = bool(entry.get("typecheck", False))
        typechecks.append(tc)
        cycle = entry.get("cycle_score")
        cycle_val = float(cycle) if isinstance(cycle, (int, float)) else float("nan")
        cycles.append(cycle_val)
        tok = _counter(_tokenize(_normalize_statement(candidate)))
        tokens.append(tok)
        lengths.append(float(sum(tok.values())))
        targets.append(_statement_f1(candidate, ground_truth))

    if not texts:
        return None

    consensus_idx, consensus_scores = _consensus_index(tokens)
    consensus_candidate = texts[consensus_idx]
    consensus_tc = typechecks[consensus_idx]
    consensus_f1 = _statement_f1(consensus_candidate, ground_truth)

    consensus_margin = 0.0
    avg_pairwise = 0.0
    if len(consensus_scores) > 1:
        sorted_scores = sorted(consensus_scores, reverse=True)
        consensus_margin = sorted_scores[0] - sorted_scores[1]
        avg_pairwise = sum(consensus_scores) / len(consensus_scores)
    tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0

    finite_cycles = [c for c in cycles if math.isfinite(c)]
    fallback_cycle = min(finite_cycles) - 1.0 if finite_cycles else 0.0
    cycles = [c if math.isfinite(c) else fallback_cycle for c in cycles]
    max_cycle = max(cycles) if cycles else 0.0
    best_cycle_idx = int(np.argmax(np.asarray(cycles)))
    best_cycle_tc = 1.0 if typechecks[best_cycle_idx] else 0.0

    cycle_z = _zscore(cycles)
    len_z = _zscore(lengths)
    cons_z = _zscore(consensus_scores if consensus_scores else [0.0 for _ in texts])

    features = []
    for i in range(len(texts)):
        features.append([1.0, cycle_z[i], len_z[i], cons_z[i], 1.0 if typechecks[i] else 0.0])

    return {
        "features": np.asarray(features, dtype=np.float64),
        "targets": np.asarray(targets, dtype=np.float64),
        "typechecks": np.asarray(typechecks, dtype=bool),
        "consensus_f1": consensus_f1,
        "consensus_tc": consensus_tc,
        "consensus_margin": consensus_margin,
        "avg_pairwise": avg_pairwise,
        "tc_rate": tc_rate,
        "max_cycle": max_cycle,
        "best_cycle_tc": best_cycle_tc,
    }


def _fit_reranker(records: List[dict], ridge: float) -> np.ndarray:
    X = []
    y = []
    for rec in records:
        X.append(rec["features"])
        y.append(rec["targets"])
    X_train = np.vstack(X)
    y_train = np.concatenate(y)
    return _fit_ridge(X_train, y_train, ridge)


def _apply_reranker(rec: dict, model: np.ndarray) -> dict:
    scores = rec["features"] @ model
    best_idx = int(np.argmax(scores))
    best_f1 = float(rec["targets"][best_idx])
    best_tc = bool(rec["typechecks"][best_idx])
    if scores.shape[0] > 1:
        sorted_scores = np.sort(scores)[::-1]
        gap = float(sorted_scores[0] - sorted_scores[1])
    else:
        gap = 0.0
    return {
        "learned_f1": best_f1,
        "learned_tc": best_tc,
        "learned_gap": gap,
        "learned_score_std": float(np.std(scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta-select consensus vs learned reranker.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--model", choices=["ridge", "gbrt"], default="ridge")
    parser.add_argument("--gbrt-estimators", type=int, default=200)
    parser.add_argument("--gbrt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbrt-max-depth", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])
    prepared = []
    for record in records:
        rec = _prepare_record(record)
        if rec is not None:
            prepared.append(rec)
    if not prepared:
        raise SystemExit("No usable records found.")

    k = max(2, args.folds)
    inner_k = max(2, args.inner_folds)
    fold_metrics = []
    weights = []
    thresholds = []
    for fold in range(k):
        train_records = []
        test_records = []
        for idx, rec in enumerate(prepared):
            if idx % k == fold:
                test_records.append(rec)
            else:
                train_records.append(rec)

        reranker_model = _fit_reranker(train_records, args.ridge)

        # Out-of-fold learned predictions for meta training.
        learned_train = [None for _ in train_records]
        for inner_fold in range(inner_k):
            inner_train = []
            inner_test_idx = []
            for idx, rec in enumerate(train_records):
                if idx % inner_k == inner_fold:
                    inner_test_idx.append(idx)
                else:
                    inner_train.append(rec)
            if not inner_train:
                continue
            inner_model = _fit_reranker(inner_train, args.ridge)
            for idx in inner_test_idx:
                learned_train[idx] = _apply_reranker(train_records[idx], inner_model)

        # Fallback: any missing inner predictions use the outer model.
        for idx, rec in enumerate(train_records):
            if learned_train[idx] is None:
                learned_train[idx] = _apply_reranker(rec, reranker_model)

        learned_test = [_apply_reranker(rec, reranker_model) for rec in test_records]

        # Build meta training data.
        meta_features = []
        meta_targets = []
        for rec, learned in zip(train_records, learned_train):
            features = [
                1.0,
                rec["consensus_margin"],
                rec["avg_pairwise"],
                rec["tc_rate"],
                rec["max_cycle"],
                rec["best_cycle_tc"],
                learned["learned_gap"],
                learned["learned_score_std"],
                1.0 if learned["learned_tc"] else 0.0,
                1.0 if rec["consensus_tc"] else 0.0,
            ]
            meta_features.append(features)
            meta_targets.append(learned["learned_f1"] - rec["consensus_f1"])

        X_train = np.asarray(meta_features, dtype=np.float64)
        y_train = np.asarray(meta_targets, dtype=np.float64)
        X_train_z = X_train.copy()
        mean = X_train_z[:, 1:].mean(axis=0)
        std = X_train_z[:, 1:].std(axis=0)
        std[std == 0] = 1.0
        X_train_z[:, 1:] = (X_train_z[:, 1:] - mean) / std

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

        preds = _predict_scores(model, X_train_z, model_type=args.model)
        candidates = np.quantile(preds, np.linspace(0.0, 1.0, 21))
        best_thr = 0.0
        best_f1 = float("-inf")
        for thr in candidates:
            f1_sum = 0.0
            for pred, rec, learned in zip(preds, train_records, learned_train):
                use_learned = pred > thr
                f1_sum += learned["learned_f1"] if use_learned else rec["consensus_f1"]
            avg = f1_sum / len(train_records)
            if avg > best_f1:
                best_f1 = avg
                best_thr = float(thr)
        thresholds.append(best_thr)

        metrics = {
            "count": 0,
            "meta_f1_sum": 0.0,
            "consensus_f1_sum": 0.0,
            "learned_f1_sum": 0.0,
            "meta_tc_sum": 0.0,
            "choose_learned_sum": 0,
        }
        for rec, learned in zip(test_records, learned_test):
            features = [
                1.0,
                rec["consensus_margin"],
                rec["avg_pairwise"],
                rec["tc_rate"],
                rec["max_cycle"],
                rec["best_cycle_tc"],
                learned["learned_gap"],
                learned["learned_score_std"],
                1.0 if learned["learned_tc"] else 0.0,
                1.0 if rec["consensus_tc"] else 0.0,
            ]
            X_test = np.asarray(features, dtype=np.float64)
            X_test[1:] = (X_test[1:] - mean) / std
            pred = float(_predict_scores(model, X_test[None, :], model_type=args.model)[0])
            use_learned = pred > best_thr
            meta_f1 = learned["learned_f1"] if use_learned else rec["consensus_f1"]
            meta_tc = learned["learned_tc"] if use_learned else rec["consensus_tc"]
            metrics["count"] += 1
            metrics["meta_f1_sum"] += meta_f1
            metrics["consensus_f1_sum"] += rec["consensus_f1"]
            metrics["learned_f1_sum"] += learned["learned_f1"]
            metrics["meta_tc_sum"] += int(bool(meta_tc))
            metrics["choose_learned_sum"] += int(bool(use_learned))
        fold_metrics.append(metrics)

    total = sum(m["count"] for m in fold_metrics)
    summary = {
        "count": total,
        "meta_f1_avg": sum(m["meta_f1_sum"] for m in fold_metrics) / total,
        "consensus_f1_avg": sum(m["consensus_f1_sum"] for m in fold_metrics) / total,
        "learned_f1_avg": sum(m["learned_f1_sum"] for m in fold_metrics) / total,
        "meta_typecheck_rate": sum(m["meta_tc_sum"] for m in fold_metrics) / total,
        "choose_learned_rate": sum(m["choose_learned_sum"] for m in fold_metrics) / total,
        "avg_threshold": float(np.mean(thresholds)),
    }
    if args.model == "ridge":
        summary["avg_weights"] = np.mean(np.asarray(weights), axis=0).tolist()
    else:
        summary["avg_feature_importances"] = np.mean(np.asarray(weights), axis=0).tolist()

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results),
                "folds": args.folds,
                "ridge": args.ridge,
                "model": args.model,
                "gbrt_estimators": args.gbrt_estimators,
                "gbrt_learning_rate": args.gbrt_learning_rate,
                "gbrt_max_depth": args.gbrt_max_depth,
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
