#!/usr/bin/env python3
"""Gate between two rerankers using a lightweight classifier on record features."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from analysis_learned_reranker import _fit_ridge, _prepare_record, _normalize_statement


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[^\s]")
NUM_RE = re.compile(r"(?<![A-Za-z_])\d+(?![A-Za-z_])")
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
WORD_NUM_RE = re.compile(r"\\b(" + "|".join(WORD_NUMBERS.keys()) + r")\\b", re.IGNORECASE)


@dataclass
class ModelConfig:
    use_cycle: bool
    extra_candidate_features: bool
    nl_number_missing_extra: bool


@dataclass
class PreparedRecord:
    features: np.ndarray
    targets: np.ndarray
    baseline_f1: float
    gate_features: np.ndarray


def _nl_number_counter(text: str) -> Dict[str, int]:
    digits = NUM_RE.findall(text)
    words = [WORD_NUMBERS[word.lower()] for word in WORD_NUM_RE.findall(text)]
    counts: Dict[str, int] = {}
    for num in digits + words:
        counts[num] = counts.get(num, 0) + 1
    return counts


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def _counter(tokens: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def _f1_score(pred: Dict[str, int], gold: Dict[str, int]) -> float:
    if not pred or not gold:
        return 0.0
    common = 0
    for key, val in pred.items():
        if key in gold:
            common += min(val, gold[key])
    if common == 0:
        return 0.0
    precision = common / sum(pred.values())
    recall = common / sum(gold.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _consensus_scores(token_counters: List[Dict[str, int]]) -> List[float]:
    if len(token_counters) == 1:
        return [0.0]
    scores = []
    for i, tok in enumerate(token_counters):
        total = 0.0
        for j, other in enumerate(token_counters):
            if i == j:
                continue
            total += _f1_score(tok, other)
        scores.append(total / max(1, len(token_counters) - 1))
    return scores


def _record_gate_features(record: dict) -> np.ndarray:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return np.zeros(6, dtype=np.float64)
    tokens = []
    for entry in candidates:
        cand = entry.get("candidate")
        if not isinstance(cand, str):
            continue
        stmt = _normalize_statement(cand)
        tokens.append(_counter(_tokenize(stmt)))
    if not tokens:
        return np.zeros(6, dtype=np.float64)

    consensus = _consensus_scores(tokens)
    sorted_scores = sorted(consensus, reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    avg_pairwise = sum(consensus) / len(consensus)

    typechecks = [bool(entry.get("typecheck", False)) for entry in candidates if isinstance(entry, dict)]
    tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0

    nl = record.get("nl_statement", "") or ""
    nl_nums = _nl_number_counter(nl)
    nl_num_total = sum(nl_nums.values())
    candidate_counts = [len(NUM_RE.findall(_normalize_statement(entry.get("candidate", "") or ""))) for entry in candidates]
    cand_num_avg = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0.0

    return np.asarray([margin, avg_pairwise, tc_rate, float(nl_num_total), cand_num_avg, float(len(tokens))])


def _prepare_records(records: List[dict], config: ModelConfig) -> List[PreparedRecord]:
    prepared: List[PreparedRecord] = []
    for record in records:
        features, targets, _typechecks, baseline_f1 = _prepare_record(
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
            include_nl_number_detail=False,
            include_nl_number_missing_extra=config.nl_number_missing_extra,
            include_nl_number_missing_extra_ratio=False,
            include_nl_number_missing_extra_signed=False,
            embedder=None,
        )
        gate_features = _record_gate_features(record)
        if features:
            feat_arr = np.asarray(features, dtype=np.float64)
            targ_arr = np.asarray(targets, dtype=np.float64)
        else:
            feat_arr = np.zeros((0, 1), dtype=np.float64)
            targ_arr = np.zeros((0,), dtype=np.float64)
        prepared.append(
            PreparedRecord(
                features=feat_arr,
                targets=targ_arr,
                baseline_f1=baseline_f1,
                gate_features=gate_features,
            )
        )
    return prepared


def _fit_reranker(prepared: List[PreparedRecord], indices: List[int], ridge: float) -> np.ndarray:
    X_train: List[np.ndarray] = []
    y_train: List[np.ndarray] = []
    for idx in indices:
        record = prepared[idx]
        if record.features.size == 0:
            continue
        X_train.append(record.features)
        y_train.append(record.targets)
    if not X_train:
        return np.zeros(1, dtype=np.float64)
    X = np.vstack(X_train)
    y = np.concatenate(y_train)
    return _fit_ridge(X, y, ridge)


def _score_record(prepared: PreparedRecord, model: np.ndarray) -> float:
    if prepared.features.size == 0:
        return 0.0
    scores = prepared.features @ model
    return prepared.targets[int(np.argmax(scores))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate between two rerankers with a classifier.")
    parser.add_argument("--results", type=Path, required=True, help="Best-of-cycle JSON file")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.results.read_text())
    records = data.get("records", [])
    if not records:
        raise SystemExit("No records found.")

    base_config = ModelConfig(use_cycle=False, extra_candidate_features=True, nl_number_missing_extra=False)
    nl_config = ModelConfig(use_cycle=False, extra_candidate_features=True, nl_number_missing_extra=True)

    k = max(2, args.folds)
    metrics = {
        "count": 0,
        "baseline_sum": 0.0,
        "base_sum": 0.0,
        "nl_sum": 0.0,
        "gate_sum": 0.0,
        "gate_choose_nl": 0,
    }

    base_prepared = _prepare_records(records, base_config)
    nl_prepared = _prepare_records(records, nl_config)

    for fold in range(k):
        train_indices = [idx for idx in range(len(records)) if idx % k != fold]
        test_indices = [idx for idx in range(len(records)) if idx % k == fold]

        base_model = _fit_reranker(base_prepared, train_indices, args.ridge)
        nl_model = _fit_reranker(nl_prepared, train_indices, args.ridge)

        gate_X = []
        gate_y = []
        for idx in train_indices:
            base_f1 = _score_record(base_prepared[idx], base_model)
            nl_f1 = _score_record(nl_prepared[idx], nl_model)
            gate_X.append(base_prepared[idx].gate_features)
            gate_y.append(1 if nl_f1 > base_f1 else 0)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(np.asarray(gate_X, dtype=np.float64), np.asarray(gate_y, dtype=np.int32))

        for idx in test_indices:
            baseline = base_prepared[idx].baseline_f1
            base_f1 = _score_record(base_prepared[idx], base_model)
            nl_f1 = _score_record(nl_prepared[idx], nl_model)
            gate_feat = base_prepared[idx].gate_features.reshape(1, -1)
            choose_nl = bool(clf.predict(gate_feat)[0])
            gate_f1 = nl_f1 if choose_nl else base_f1

            metrics["count"] += 1
            metrics["baseline_sum"] += baseline
            metrics["base_sum"] += base_f1
            metrics["nl_sum"] += nl_f1
            metrics["gate_sum"] += gate_f1
            metrics["gate_choose_nl"] += int(choose_nl)

    summary = {
        "count": metrics["count"],
        "baseline_f1_avg": metrics["baseline_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "base_f1_avg": metrics["base_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "nl_f1_avg": metrics["nl_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "gate_f1_avg": metrics["gate_sum"] / metrics["count"] if metrics["count"] else 0.0,
        "gate_choose_nl_rate": metrics["gate_choose_nl"] / metrics["count"] if metrics["count"] else 0.0,
    }

    if args.output is not None:
        payload = {"meta": {"results_path": str(args.results), "folds": args.folds}, "summary": summary}
        args.output.write_text(json.dumps(payload, indent=2))

    print("Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
