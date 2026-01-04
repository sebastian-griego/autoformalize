#!/usr/bin/env python3
"""Train a reranker on candidate features with k-fold CV."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


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
WORD_NUM_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS.keys()) + r")\b", re.IGNORECASE)
DEFAULT_LENGTH_PENALTY = 0.001


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


def _extract_nl_numbers(text: str) -> List[str]:
    if not text:
        return []
    digits = NUM_RE.findall(text)
    words = [WORD_NUMBERS[word.lower()] for word in WORD_NUM_RE.findall(text)]
    return digits + words


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


def _zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def _rank_fraction(values: List[float]) -> List[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2.0
        frac = 1.0 - (avg_rank / (len(indexed) - 1))
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = frac
        i = j + 1
    return ranks


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


class _Embedder:
    def __init__(self, model_id: str, device: str | None, batch_size: int, max_length: int) -> None:
        self.model_id = model_id
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.cache: Dict[str, torch.Tensor] = {}

    def embed(self, texts: List[str]) -> List[torch.Tensor]:
        missing = [text for text in texts if text not in self.cache]
        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                for text, emb in zip(batch, pooled.cpu()):
                    self.cache[text] = emb
        return [self.cache[text] for text in texts]


def _embed_consensus_scores(texts: List[str], embedder: _Embedder) -> List[float]:
    if len(texts) <= 1:
        return [0.0 for _ in texts]
    embeddings = embedder.embed(texts)
    emb = torch.stack(embeddings, dim=0)
    sim = emb @ emb.T
    n = sim.size(0)
    avg = (sim.sum(dim=1) - 1.0) / max(1, n - 1)
    return avg.tolist()


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


def _prepare_record(
    record: dict,
    *,
    include_baseline: bool,
    include_forward: bool,
    include_cycle: bool,
    include_length: bool,
    include_consensus: bool,
    include_typecheck: bool,
    include_embed_consensus: bool,
    include_context_interactions: bool,
    include_extra_candidate_features: bool,
    include_numeric_features: bool,
    include_nl_number_features: bool,
    include_nl_number_detail: bool = False,
    include_nl_number_missing_extra: bool = False,
    include_nl_number_missing_extra_ratio: bool = False,
    include_nl_number_missing_extra_signed: bool = False,
    embedder: _Embedder | None,
) -> Tuple[List[List[float]], List[float], List[bool], float]:
    candidates = record.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return [], [], [], 0.0
    ground_truth = record.get("ground_truth", "")
    nl_statement = record.get("nl_statement", "") or ""
    baseline_candidate = record.get("baseline_candidate", "")
    baseline_typecheck = bool(record.get("baseline_typecheck", False))

    entries = list(candidates)
    baseline_forward = record.get("baseline_forward_score")
    if baseline_candidate and not any(
        isinstance(entry, dict) and entry.get("candidate") == baseline_candidate
        for entry in entries
    ):
        entry = {"candidate": baseline_candidate, "typecheck": baseline_typecheck}
        if include_forward:
            entry["forward_score"] = baseline_forward
        entries.append(entry)

    candidates_text: List[str] = []
    cycles: List[float] = []
    lengths: List[float] = []
    lenpen_scores: List[float] = []
    typechecks: List[bool] = []
    token_counters: List[Counter[str]] = []
    forward_scores: List[float] = []
    normalized_texts: List[str] = []
    num_counts: List[float] = []
    num_ratios: List[float] = []
    nl_num_f1s: List[float] = []
    nl_num_precisions: List[float] = []
    nl_num_recalls: List[float] = []
    nl_num_abs_deltas: List[float] = []
    nl_num_missing: List[float] = []
    nl_num_extra: List[float] = []
    nl_num_missing_ratio: List[float] = []
    nl_num_extra_ratio: List[float] = []
    nl_num_signed_delta: List[float] = []

    finite_cycles: List[float] = []
    finite_forward: List[float] = []
    nl_num_counter = Counter(_extract_nl_numbers(nl_statement))
    nl_num_total = sum(nl_num_counter.values())
    for entry in entries:
        candidate = entry.get("candidate")
        if not isinstance(candidate, str):
            continue
        cycle = entry.get("cycle_score")
        cycle_val = float(cycle) if isinstance(cycle, (int, float)) else float("nan")
        if math.isfinite(cycle_val):
            finite_cycles.append(cycle_val)
        if include_forward:
            forward = entry.get("forward_score")
            forward_val = float(forward) if isinstance(forward, (int, float)) else float("nan")
            if math.isfinite(forward_val):
                finite_forward.append(forward_val)
        candidates_text.append(candidate)
        cycles.append(cycle_val)
        statement = _normalize_statement(candidate)
        normalized_texts.append(statement)
        tokens = _tokenize(statement)
        lengths.append(float(len(tokens)))
        num_count = float(len(NUM_RE.findall(statement)))
        num_counts.append(num_count)
        num_ratios.append(num_count / max(1.0, float(len(tokens))))
        if (
            include_nl_number_features
            or include_nl_number_detail
            or include_nl_number_missing_extra
            or include_nl_number_missing_extra_ratio
            or include_nl_number_missing_extra_signed
        ):
            cand_num_counter = Counter(NUM_RE.findall(statement))
            cand_num_total = sum(cand_num_counter.values())
            common = sum((cand_num_counter & nl_num_counter).values()) if nl_num_counter else 0
            if include_nl_number_features:
                if nl_num_counter:
                    nl_num_f1s.append(_f1_score(cand_num_counter, nl_num_counter))
                else:
                    nl_num_f1s.append(0.0)
            if include_nl_number_detail:
                if nl_num_counter:
                    precision = common / cand_num_total if cand_num_total else 0.0
                    recall = common / nl_num_total if nl_num_total else 0.0
                    nl_num_precisions.append(precision)
                    nl_num_recalls.append(recall)
                    nl_num_abs_deltas.append(abs(cand_num_total - nl_num_total))
                else:
                    nl_num_precisions.append(0.0)
                    nl_num_recalls.append(0.0)
                    nl_num_abs_deltas.append(float(cand_num_total))
            if include_nl_number_missing_extra:
                missing = float(nl_num_total - common) if nl_num_counter else 0.0
                extra = float(cand_num_total - common)
                nl_num_missing.append(missing)
                nl_num_extra.append(extra)
            if include_nl_number_missing_extra_ratio:
                if nl_num_total:
                    missing_ratio = float(nl_num_total - common) / float(nl_num_total)
                    extra_ratio = float(cand_num_total - common) / float(nl_num_total)
                else:
                    missing_ratio = 0.0
                    extra_ratio = 0.0
                nl_num_missing_ratio.append(missing_ratio)
                nl_num_extra_ratio.append(extra_ratio)
            if include_nl_number_missing_extra_signed:
                if nl_num_total:
                    signed_delta = float(cand_num_total - nl_num_total) / float(nl_num_total)
                else:
                    signed_delta = 0.0
                nl_num_signed_delta.append(signed_delta)
        else:
            if include_nl_number_features:
                nl_num_f1s.append(0.0)
            if include_nl_number_detail:
                nl_num_precisions.append(0.0)
                nl_num_recalls.append(0.0)
                nl_num_abs_deltas.append(0.0)
            if include_nl_number_missing_extra:
                nl_num_missing.append(0.0)
                nl_num_extra.append(0.0)
            if include_nl_number_missing_extra_ratio:
                nl_num_missing_ratio.append(0.0)
                nl_num_extra_ratio.append(0.0)
            if include_nl_number_missing_extra_signed:
                nl_num_signed_delta.append(0.0)
        lenpen_val = entry.get("length_penalized_score")
        if isinstance(lenpen_val, (int, float)) and math.isfinite(lenpen_val):
            lenpen_scores.append(float(lenpen_val))
        else:
            lenpen_scores.append(cycle_val - DEFAULT_LENGTH_PENALTY * float(len(tokens)))
        token_counters.append(_counter(tokens))
        typechecks.append(bool(entry.get("typecheck", False)))
        if include_forward:
            forward_scores.append(forward_val)

    if not candidates_text:
        return [], [], [], 0.0

    fallback_cycle = min(finite_cycles) - 1.0 if finite_cycles else 0.0
    cycles = [c if math.isfinite(c) else fallback_cycle for c in cycles]
    if include_forward:
        fallback_forward = min(finite_forward) - 1.0 if finite_forward else 0.0
        forward_scores = [f if math.isfinite(f) else fallback_forward for f in forward_scores]
    consensus = _consensus_scores(token_counters)
    tc_rate = sum(typechecks) / len(typechecks) if typechecks else 0.0
    consensus_margin = 0.0
    avg_pairwise = 0.0
    if len(consensus) > 1:
        sorted_scores = sorted(consensus, reverse=True)
        consensus_margin = sorted_scores[0] - sorted_scores[1]
        avg_pairwise = sum(consensus) / len(consensus)
    baseline_tokens = _counter(_tokenize(_normalize_statement(baseline_candidate)))
    baseline_sim = [_f1_score(tok, baseline_tokens) for tok in token_counters]

    cycle_z = _zscore(cycles)
    len_z = _zscore(lengths)
    cons_z = _zscore(consensus)
    embed_z = _zscore(_embed_consensus_scores(normalized_texts, embedder)) if include_embed_consensus else []
    lenpen_z = _zscore(lenpen_scores)
    num_count_z = _zscore(num_counts)
    num_ratio_z = _zscore(num_ratios)
    nl_num_f1_z = _zscore(nl_num_f1s) if include_nl_number_features else []
    nl_num_prec_z = _zscore(nl_num_precisions) if include_nl_number_detail else []
    nl_num_rec_z = _zscore(nl_num_recalls) if include_nl_number_detail else []
    nl_num_abs_delta_z = _zscore(nl_num_abs_deltas) if include_nl_number_detail else []
    nl_num_missing_z = _zscore(nl_num_missing) if include_nl_number_missing_extra else []
    nl_num_extra_z = _zscore(nl_num_extra) if include_nl_number_missing_extra else []
    nl_num_missing_ratio_z = (
        _zscore(nl_num_missing_ratio) if include_nl_number_missing_extra_ratio else []
    )
    nl_num_extra_ratio_z = (
        _zscore(nl_num_extra_ratio) if include_nl_number_missing_extra_ratio else []
    )
    nl_num_signed_delta_z = (
        _zscore(nl_num_signed_delta) if include_nl_number_missing_extra_signed else []
    )
    cycle_rank = _rank_fraction(cycles)
    len_rank = _rank_fraction(lengths)
    cons_rank = _rank_fraction(consensus)
    base_z = _zscore(baseline_sim)
    forward_z = _zscore(forward_scores) if include_forward else []

    tc_indices = [idx for idx, tc in enumerate(typechecks) if tc]
    tc_consensus: List[float] = []
    if tc_indices:
        for i, tok in enumerate(token_counters):
            total = 0.0
            denom = 0
            for j in tc_indices:
                if i == j:
                    continue
                total += _f1_score(tok, token_counters[j])
                denom += 1
            tc_consensus.append(total / denom if denom else 0.0)
    else:
        tc_consensus = [0.0 for _ in token_counters]
    tc_cons_z = _zscore(tc_consensus)
    tc_cons_rank = _rank_fraction(tc_consensus)

    features: List[List[float]] = []
    targets: List[float] = []
    for i, candidate in enumerate(candidates_text):
        row = [1.0]
        cycle_feat = cycle_z[i]
        len_feat = len_z[i]
        cons_feat = cons_z[i]
        tc_feat = 1.0 if typechecks[i] else 0.0
        embed_feat = embed_z[i] if include_embed_consensus else 0.0
        if include_cycle:
            row.append(cycle_feat)
        if include_length:
            row.append(len_feat)
        if include_consensus:
            row.append(cons_feat)
        if include_embed_consensus:
            row.append(embed_feat)
        if include_typecheck:
            row.append(tc_feat)
        if include_context_interactions:
            row.extend(
                [
                    cycle_feat * tc_rate,
                    cons_feat * tc_rate,
                    tc_feat * tc_rate,
                    cycle_feat * consensus_margin,
                    cons_feat * consensus_margin,
                    cycle_feat * avg_pairwise,
                    cons_feat * avg_pairwise,
                ]
            )
        if include_extra_candidate_features:
            row.extend(
                [
                    lenpen_z[i],
                    cycle_rank[i],
                    len_rank[i],
                    cons_rank[i],
                    tc_cons_z[i],
                    tc_cons_rank[i],
                ]
            )
        if include_numeric_features:
            row.extend([num_count_z[i], num_ratio_z[i]])
        if include_nl_number_features:
            row.append(nl_num_f1_z[i])
        if include_nl_number_detail:
            row.extend([nl_num_prec_z[i], nl_num_rec_z[i], nl_num_abs_delta_z[i]])
        if include_nl_number_missing_extra:
            row.extend([nl_num_missing_z[i], nl_num_extra_z[i]])
        if include_nl_number_missing_extra_ratio:
            row.extend([nl_num_missing_ratio_z[i], nl_num_extra_ratio_z[i]])
        if include_nl_number_missing_extra_signed:
            row.append(nl_num_signed_delta_z[i])
        if include_forward:
            row.append(forward_z[i])
        if include_baseline:
            row.append(base_z[i])
        features.append(row)
        targets.append(_statement_f1(candidate, ground_truth))

    baseline_f1 = _statement_f1(baseline_candidate, ground_truth)
    return features, targets, typechecks, baseline_f1


def _feature_names(
    *,
    include_cycle: bool,
    include_length: bool,
    include_consensus: bool,
    include_embed_consensus: bool,
    include_typecheck: bool,
    include_context_interactions: bool,
    include_extra_candidate_features: bool,
    include_numeric_features: bool,
    include_nl_number_features: bool,
    include_nl_number_detail: bool,
    include_nl_number_missing_extra: bool,
    include_nl_number_missing_extra_ratio: bool,
    include_nl_number_missing_extra_signed: bool,
    include_forward: bool,
    include_baseline: bool,
) -> List[str]:
    names = ["bias"]
    if include_cycle:
        names.append("cycle_z")
    if include_length:
        names.append("len_z")
    if include_consensus:
        names.append("cons_z")
    if include_embed_consensus:
        names.append("embed_cons_z")
    if include_typecheck:
        names.append("typecheck")
    if include_context_interactions:
        names.extend(
            [
                "cycle_tc_rate",
                "cons_tc_rate",
                "typecheck_tc_rate",
                "cycle_cons_margin",
                "cons_cons_margin",
                "cycle_avg_pairwise",
                "cons_avg_pairwise",
            ]
        )
    if include_extra_candidate_features:
        names.extend(
            [
                "lenpen_z",
                "cycle_rank",
                "len_rank",
                "cons_rank",
                "tc_cons_z",
                "tc_cons_rank",
            ]
        )
    if include_numeric_features:
        names.extend(["num_count_z", "num_ratio_z"])
    if include_nl_number_features:
        names.append("nl_num_f1_z")
    if include_nl_number_detail:
        names.extend(["nl_num_prec_z", "nl_num_rec_z", "nl_num_abs_delta_z"])
    if include_nl_number_missing_extra:
        names.extend(["nl_num_missing_z", "nl_num_extra_z"])
    if include_nl_number_missing_extra_ratio:
        names.extend(["nl_num_missing_ratio_z", "nl_num_extra_ratio_z"])
    if include_nl_number_missing_extra_signed:
        names.append("nl_num_signed_delta_z")
    if include_forward:
        names.append("forward_z")
    if include_baseline:
        names.append("baseline_sim_z")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a reranker with k-fold CV or train/test split.")
    parser.add_argument("--results", type=Path, default=None, help="Best-of-cycle JSON file for CV")
    parser.add_argument("--train-results", type=Path, default=None, help="Training JSON file for train/test mode")
    parser.add_argument("--test-results", type=Path, default=None, help="Test JSON file for train/test mode")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", choices=["ridge", "gbrt"], default="ridge")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--gbrt-estimators", type=int, default=200)
    parser.add_argument("--gbrt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbrt-depth", type=int, default=3)
    parser.add_argument("--baseline-feature", action="store_true", help="Include baseline similarity feature")
    parser.add_argument("--forward-feature", action="store_true", help="Include forward LM score feature")
    parser.add_argument("--embed-consensus", action="store_true", help="Include embedding consensus feature")
    parser.add_argument("--embed-model-id", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--embed-max-length", type=int, default=256)
    parser.add_argument("--embed-device", type=str, default=None)
    parser.add_argument("--context-interactions", action="store_true", help="Add consensus/tc interaction features")
    parser.add_argument("--extra-candidate-features", action="store_true", help="Add per-candidate stats features")
    parser.add_argument("--numeric-features", action="store_true", help="Add numeric literal features")
    parser.add_argument("--nl-number-features", action="store_true", help="Add NL-number overlap features")
    parser.add_argument(
        "--nl-number-detail",
        action="store_true",
        help="Add detailed NL-number precision/recall features",
    )
    parser.add_argument(
        "--nl-number-missing-extra",
        action="store_true",
        help="Add NL-number missing/extra count features",
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
    parser.add_argument("--no-cycle", action="store_true", help="Drop cycle-score feature")
    parser.add_argument("--no-length", action="store_true", help="Drop length feature")
    parser.add_argument("--no-consensus", action="store_true", help="Drop consensus feature")
    parser.add_argument("--no-typecheck", action="store_true", help="Drop typecheck feature")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    embedder = None
    if args.embed_consensus:
        embedder = _Embedder(
            args.embed_model_id,
            device=args.embed_device,
            batch_size=args.embed_batch_size,
            max_length=args.embed_max_length,
        )
    include_nl_number_features = args.nl_number_features or args.nl_number_detail

    def _load_prepared(path: Path) -> List[Tuple[List[List[float]], List[float], List[bool], float]]:
        data = json.loads(path.read_text())
        records = data.get("records", [])
        prepared: List[Tuple[List[List[float]], List[float], List[bool], float]] = []
        for record in records:
            features, targets, typechecks, baseline_f1 = _prepare_record(
                record,
                include_baseline=args.baseline_feature,
                include_forward=args.forward_feature,
                include_cycle=not args.no_cycle,
                include_length=not args.no_length,
                include_consensus=not args.no_consensus,
                include_typecheck=not args.no_typecheck,
                include_embed_consensus=args.embed_consensus,
                include_context_interactions=args.context_interactions,
                include_extra_candidate_features=args.extra_candidate_features,
                include_numeric_features=args.numeric_features,
                include_nl_number_features=include_nl_number_features,
                include_nl_number_detail=args.nl_number_detail,
                include_nl_number_missing_extra=args.nl_number_missing_extra,
                include_nl_number_missing_extra_ratio=args.nl_number_missing_extra_ratio,
                include_nl_number_missing_extra_signed=args.nl_number_missing_extra_signed,
                embedder=embedder,
            )
            if features:
                prepared.append((features, targets, typechecks, baseline_f1))
        return prepared

    if args.train_results is not None:
        train_path = args.train_results
        test_path = args.test_results or args.train_results
        train_prepared = _load_prepared(train_path)
        test_prepared = _load_prepared(test_path)
        if not train_prepared or not test_prepared:
            raise SystemExit("No usable records found for train/test mode.")
        X_train = []
        y_train = []
        for features, targets, _typechecks, _baseline_f1 in train_prepared:
            X_train.extend(features)
            y_train.extend(targets)
        X_train_np = np.asarray(X_train, dtype=np.float64)
        y_train_np = np.asarray(y_train, dtype=np.float64)
        gbrt_params = {
            "n_estimators": args.gbrt_estimators,
            "learning_rate": args.gbrt_learning_rate,
            "max_depth": args.gbrt_depth,
        }
        model = _fit_model(X_train_np, y_train_np, model=args.model, ridge=args.ridge, gbrt_params=gbrt_params)
        weights = []
        importances = []
        if args.model == "ridge":
            weights.append(model.tolist())
        elif args.model == "gbrt":
            importances.append(model.feature_importances_.tolist())

        fold_metrics = []
        metrics = {
            "count": 0,
            "baseline_f1_sum": 0.0,
            "learned_f1_sum": 0.0,
            "learned_tc_f1_sum": 0.0,
            "learned_typecheck_sum": 0.0,
            "learned_tc_typecheck_sum": 0.0,
            "learned_improve_sum": 0,
            "learned_tc_improve_sum": 0,
        }
        for features, targets, typechecks, baseline_f1 in test_prepared:
            scores = _predict_scores(model, np.asarray(features, dtype=np.float64), model_type=args.model)
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
            metrics["learned_improve_sum"] += int(best_f1 > baseline_f1)
            metrics["learned_tc_improve_sum"] += int(best_tc_f1 > baseline_f1)
        fold_metrics.append(metrics)
    else:
        if args.results is None:
            raise SystemExit("Provide --results for CV mode or --train-results for train/test mode.")
        prepared = _load_prepared(args.results)
        if not prepared:
            raise SystemExit("No usable records found.")

        k = max(2, args.folds)
        fold_metrics = []
        weights = []
        importances = []
        gbrt_params = {
            "n_estimators": args.gbrt_estimators,
            "learning_rate": args.gbrt_learning_rate,
            "max_depth": args.gbrt_depth,
        }
        for fold in range(k):
            X_train: List[List[float]] = []
            y_train: List[float] = []
            test_records = []
            for idx, item in enumerate(prepared):
                if idx % k == fold:
                    test_records.append(item)
                else:
                    X_train.extend(item[0])
                    y_train.extend(item[1])
            X_train_np = np.asarray(X_train, dtype=np.float64)
            y_train_np = np.asarray(y_train, dtype=np.float64)
            model = _fit_model(
                X_train_np,
                y_train_np,
                model=args.model,
                ridge=args.ridge,
                gbrt_params=gbrt_params,
            )
            if args.model == "ridge":
                weights.append(model.tolist())
            elif args.model == "gbrt":
                importances.append(model.feature_importances_.tolist())

            metrics = {
                "count": 0,
                "baseline_f1_sum": 0.0,
                "learned_f1_sum": 0.0,
                "learned_tc_f1_sum": 0.0,
                "learned_typecheck_sum": 0.0,
                "learned_tc_typecheck_sum": 0.0,
                "learned_improve_sum": 0,
                "learned_tc_improve_sum": 0,
            }
            for features, targets, typechecks, baseline_f1 in test_records:
                scores = _predict_scores(model, np.asarray(features, dtype=np.float64), model_type=args.model)
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
                metrics["learned_improve_sum"] += int(best_f1 > baseline_f1)
                metrics["learned_tc_improve_sum"] += int(best_tc_f1 > baseline_f1)

            fold_metrics.append(metrics)

    total_count = sum(m["count"] for m in fold_metrics)
    feature_names = _feature_names(
        include_cycle=not args.no_cycle,
        include_length=not args.no_length,
        include_consensus=not args.no_consensus,
        include_embed_consensus=args.embed_consensus,
        include_typecheck=not args.no_typecheck,
        include_context_interactions=args.context_interactions,
        include_extra_candidate_features=args.extra_candidate_features,
        include_numeric_features=args.numeric_features,
        include_nl_number_features=include_nl_number_features,
        include_nl_number_detail=args.nl_number_detail,
        include_nl_number_missing_extra=args.nl_number_missing_extra,
        include_nl_number_missing_extra_ratio=args.nl_number_missing_extra_ratio,
        include_nl_number_missing_extra_signed=args.nl_number_missing_extra_signed,
        include_forward=args.forward_feature,
        include_baseline=args.baseline_feature,
    )
    summary = {
        "count": total_count,
        "baseline_f1_avg": sum(m["baseline_f1_sum"] for m in fold_metrics) / total_count,
        "learned_f1_avg": sum(m["learned_f1_sum"] for m in fold_metrics) / total_count,
        "learned_tc_f1_avg": sum(m["learned_tc_f1_sum"] for m in fold_metrics) / total_count,
        "learned_typecheck_rate": sum(m["learned_typecheck_sum"] for m in fold_metrics) / total_count,
        "learned_tc_typecheck_rate": sum(m["learned_tc_typecheck_sum"] for m in fold_metrics) / total_count,
        "learned_improve_rate": sum(m["learned_improve_sum"] for m in fold_metrics) / total_count,
        "learned_tc_improve_rate": sum(m["learned_tc_improve_sum"] for m in fold_metrics) / total_count,
        "feature_names": feature_names,
    }
    if args.model == "ridge":
        summary["avg_weights"] = np.mean(np.asarray(weights), axis=0).tolist()
    elif args.model == "gbrt":
        summary["avg_feature_importances"] = np.mean(np.asarray(importances), axis=0).tolist()

    if args.output is not None:
        payload = {
            "meta": {
                "results_path": str(args.results) if args.results else None,
                "train_results": str(args.train_results) if args.train_results else None,
                "test_results": str(args.test_results) if args.test_results else None,
                "folds": args.folds,
                "model": args.model,
                "ridge": args.ridge,
                "gbrt_estimators": args.gbrt_estimators,
                "gbrt_learning_rate": args.gbrt_learning_rate,
                "gbrt_depth": args.gbrt_depth,
                "baseline_feature": args.baseline_feature,
                "forward_feature": args.forward_feature,
                "embed_consensus": args.embed_consensus,
                "context_interactions": args.context_interactions,
                "extra_candidate_features": args.extra_candidate_features,
                "numeric_features": args.numeric_features,
                "nl_number_features": include_nl_number_features,
                "nl_number_detail": args.nl_number_detail,
                "nl_number_missing_extra": args.nl_number_missing_extra,
                "nl_number_missing_extra_ratio": args.nl_number_missing_extra_ratio,
                "nl_number_missing_extra_signed": args.nl_number_missing_extra_signed,
                "embed_model_id": args.embed_model_id if args.embed_consensus else None,
                "embed_batch_size": args.embed_batch_size if args.embed_consensus else None,
                "embed_max_length": args.embed_max_length if args.embed_consensus else None,
                "embed_device": args.embed_device if args.embed_consensus else None,
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
