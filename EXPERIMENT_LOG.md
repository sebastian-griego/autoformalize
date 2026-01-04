# Experiment Log

This log summarizes the experiments run in this repo and the main outcomes.

## Environment

- GPUs: 3x NVIDIA L40
- Key libraries: vLLM 0.10.0, torch 2.7.1+cu126, transformers 4.57.3
- Dataset: `PAug/ProofNetVerif` split `valid`

## Code changes that affect results

- Added dataset sharding flags (`--num-shards`, `--shard-id`) and indexing fixes.
- Added posterior fallback from raw particle log-weights if decoded posterior sums to zero.
- Relaxed Lean well-typed completion gate: now accepts any prefix containing `theorem` and relies on cleaning to add `:= by sorry`.
- Best-of pipeline now supports `--length-penalty`, clamps non-finite cycle scores to `-inf`, and falls back to baseline when cycle scores are missing.
- Best-of pipeline now recomputes summary stats when resuming from existing records.

## Full pipeline (SMC full)

### Qwen reference, stride 32, num_particles 1 (pre-relax)

- 30 examples (3 shards), `--max-tokens 256`, `--four-bit`
- Results (aggregate):
  - baseline expected_correct 0.500, avg runtime 3.21s, avg tokens 92.0
  - smc_both expected_correct 0.267, avg runtime 181.01s, avg tokens 92.9
  - smc_both zero-sum posteriors: 9/30

### Qwen reference, stride 32, num_particles 1 (relaxed gate)

- 30 examples (3 shards), `--max-tokens 256`, `--four-bit`
- Results (aggregate):
  - baseline expected_correct 0.433, avg runtime 3.41s, avg tokens 93.8
  - smc_both expected_correct 0.433, avg runtime 185.89s, avg tokens 92.6
  - smc_both zero-sum posteriors: 0/30
  - smc_only wins: `Rudin|exercise_3_20`
  - baseline_only wins: `Shakarchi|exercise_1_13c`

### Qwen reference, stride 64, num_particles 1 (relaxed gate)

- 30 examples (3 shards), `--max-tokens 256`, `--four-bit`
- Results (aggregate):
  - baseline expected_correct 0.467, avg runtime 3.20s, avg tokens 88.2
  - smc_both expected_correct 0.433, avg runtime 186.30s, avg tokens 91.7
  - smc_both zero-sum posteriors: 0/30
  - baseline_only wins: `Rudin|exercise_3_20`

### Qwen reference, stride 32, num_particles 2 (relaxed gate)

- 3 examples (single shard), `--max-tokens 256`, `--four-bit`
- Results:
  - baseline expected_correct 0.667, avg runtime 3.14s
  - smc_both expected_correct 0.333, avg runtime 338.24s

## Ablations (`--smc-config all`)

### Qwen reference, stride 64, num_particles 1

- 12 examples (3 shards), `--max-tokens 256`, `--four-bit`
- Results (aggregate):
  - baseline expected_correct 0.500, avg runtime 3.50s, avg tokens 99.6
  - smc_no_potential expected_correct 0.500, avg runtime 3.10s, avg tokens 89.2
  - smc_lean_only expected_correct 0.500, avg runtime 3.90s, avg tokens 92.9
  - smc_cycle_only expected_correct 0.500, avg runtime 216.06s, avg tokens 105.8
  - smc_both expected_correct 0.500, avg runtime 185.99s, avg tokens 92.9
  - zero-sum posteriors: 0 for all methods

### TinyLlama reference, stride 64, num_particles 1

- Reference model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- 12 examples (3 shards), `--max-tokens 256`, `--four-bit`
- Results (aggregate):
  - baseline expected_correct 0.417, avg runtime 3.46s, avg tokens 95.7
  - smc_no_potential expected_correct 0.500, avg runtime 3.38s, avg tokens 94.1
  - smc_lean_only expected_correct 0.500, avg runtime 4.25s, avg tokens 94.2
  - smc_cycle_only expected_correct 0.500, avg runtime 190.57s, avg tokens 94.6
  - smc_both expected_correct 0.500, avg runtime 186.65s, avg tokens 93.2
  - smc_only wins: `Shakarchi|exercise_1_13c`

## Type-check evaluation (Lean well-typedness)

- Script: `analysis_typecheck_results.py` (outputs `typecheck_*.json`).
- Qwen reference, stride 32, relaxed gate (30 examples):
  - baseline top1 typecheck 0.700; smc_both 0.700
  - BEq+ top1 true rate 0.433; BEq+ false negatives vs typecheck 0.267 (8/30)
- Qwen reference, stride 64, relaxed gate (30 examples):
  - baseline top1 typecheck 0.767; smc_both 0.833
  - BEq+ top1 true rate 0.467 (baseline), 0.433 (smc_both)
  - BEq+ false negatives vs typecheck 0.300 (baseline, 9/30), 0.400 (smc_both, 12/30)
- Ablations, Qwen reference, stride 64 (12 examples):
  - baseline top1 typecheck 0.583
  - smc_no_potential / smc_lean_only / smc_cycle_only top1 typecheck 0.667
- Ablations, TinyLlama reference, stride 64 (12 examples):
  - baseline top1 typecheck 0.583
  - smc_lean_only top1 typecheck 0.750 (best on this slice)
- No BEq+ false positives observed in these slices (beq_true & typecheck false 0.0).

## Notes

- Cycle consistency remains the runtime bottleneck (minutes per example) with little accuracy gain on these slices.
- LLAMPPL emits occasional `divide by zero` warnings; after the relaxed gate, zero-sum posteriors dropped to 0/30 in the full runs.

## Best-of-N sampling + cycle reranking (no SMC)

- New script: `run_kimina_bestof_cycle.py` (samples multiple candidates and reranks by cycle score).
- Run: Qwen reference, num_candidates 8, max_tokens 256, 30 requested examples (27 unique IDs processed).
- Summary:
  - baseline typecheck rate 0.741
  - best_cycle typecheck rate 0.704 (cycle-only selection can pick non-typechecked outputs)
  - best_cycle_tc typecheck rate 0.926 (cycle score among typechecked candidates)
  - baseline cycle avg -3.114; best_cycle avg -3.014; best_cycle_tc avg -3.045
  - best_cycle chosen candidate differed from baseline 0.741 of the time; cycle score improved 0.593 of the time
  - avg unique candidates 3.48; avg typechecked candidates 2.48
- Token-level statement similarity (proxy metric vs ground truth; `analysis_statement_similarity.py`):
  - baseline F1 0.801
  - best_cycle F1 0.807
  - best_cycle_tc F1 0.812
  - best_cycle_tc improved over baseline on 0.296 of examples

### Best-of-N with higher temperature

- Run: num_candidates 16, temperature 0.4, max_tokens 256, 20 examples.
- Summary:
  - baseline typecheck 0.750; best_cycle typecheck 0.800; best_cycle_tc typecheck 1.000
  - baseline cycle avg -3.270; best_cycle avg -3.067
  - avg unique candidates 9.7; avg typechecked candidates 6.1
- Token-level statement similarity:
  - baseline F1 0.791
  - best_cycle F1 0.801
  - best_cycle_tc F1 0.791
  - best_cycle improved over baseline on 0.45 of examples

### Best-of-N with Kimina reference

- Run: reference model `AI-MO/Kimina-Autoformalizer-7B`, num_candidates 16, temperature 0.2, 40 requested examples (35 unique).
- Summary:
  - baseline typecheck 0.743; best_cycle typecheck 0.714; best_cycle_tc typecheck 0.857
  - baseline cycle avg -1.892; best_cycle avg -1.841; best_cycle_tc avg -1.873
  - avg unique candidates 5.34; avg typechecked candidates 3.74
- Token-level statement similarity:
  - baseline F1 0.808
  - best_cycle F1 0.807
  - best_cycle_tc F1 0.807

### Best-of-N Qwen reference (more examples)

- Run: num_candidates 16, temperature 0.2, 50 requested examples (43 unique).
- Summary:
  - baseline typecheck 0.791; best_cycle typecheck 0.721; best_cycle_tc typecheck 0.884
  - baseline cycle avg -2.912; best_cycle avg -2.794; best_cycle_tc avg -2.829
  - avg unique candidates 5.44; avg typechecked candidates 4.09
- Token-level statement similarity:
  - baseline F1 0.820
  - best_cycle F1 0.827
  - best_cycle_tc F1 0.820

### Best-of-N with length penalty (larger slice)

- Run: num_candidates 16, temperature 0.2, length_penalty 0.001, 100 requested examples (67 unique).
- Summary:
  - baseline typecheck 0.821; best_cycle typecheck 0.701; best_lenpen typecheck 0.716
  - best_cycle_tc typecheck 0.881; best_lenpen_tc typecheck 0.881
  - baseline cycle avg -2.959; best_cycle avg -2.813; best_lenpen avg -2.869
  - baseline lenpen avg -3.012; best_lenpen_tc avg -2.911
- Token-level statement similarity:
  - baseline F1 0.811
  - best_cycle F1 0.803
  - best_lenpen F1 0.806
  - best_lenpen_tc F1 0.802

### Best-of-N with length penalty (start-index 100)

- Run: num_candidates 16, temperature 0.2, length_penalty 0.01, start_index 100, 100 requested (72 unique).
- Summary:
  - baseline typecheck 0.764; best_cycle typecheck 0.750; best_lenpen typecheck 0.778
  - best_cycle_tc typecheck 0.889; best_lenpen_tc typecheck 0.889
- Token-level statement similarity:
  - baseline F1 0.803
  - best_cycle F1 0.807
  - best_lenpen F1 0.797
- Rerank sweep on this slice favors alpha 0.0 (all-F1 0.807); length penalty does not help here.

### Best-of-N with more candidates

- Run: num_candidates 32, temperature 0.2, 20 examples.
- Summary:
  - baseline typecheck 0.850; best_cycle typecheck 0.800; best_cycle_tc typecheck 1.000
  - baseline cycle avg -3.315; best_cycle avg -3.125; best_cycle_tc avg -3.193
  - avg unique candidates 8.2; avg typechecked candidates 4.95
- Token-level statement similarity:
  - baseline F1 0.775
  - best_cycle F1 0.782
  - best_cycle_tc F1 0.795

### Length-penalized cycle reranking

- New script: `analysis_rerank_sweep.py` (reranks by `cycle_score - alpha * len(statement_tokens)`).
- Best-of-8 (Qwen, 27 unique): best all-F1 at alpha 0.02 (0.816 vs 0.807 at alpha 0.0); best tc-F1 at alpha 0.001–0.002 (0.824 vs 0.812).
- Best-of-16 temp0.4 (Qwen, 20 examples): best all-F1 at alpha 0.005 or 0.02 (0.812 vs 0.801).
- Best-of-16 temp0.2 (Qwen, 43 unique): best all-F1 at alpha 0.001 (0.836 vs 0.827).
- Best-of-16 temp0.2 (Qwen, 67 unique): best all-F1 at alpha 0.01 (0.818 vs 0.803); best tc-F1 at alpha 0.01–0.02 (0.809 vs 0.800).
- Best-of-16 temp0.2 (Kimina ref, 35 unique): best all-F1 at alpha 0.001 (0.813 vs 0.807).
- Best-of-32 temp0.2 (Qwen, 20 examples): best all-F1 at alpha 0.02 (0.795 vs 0.782); best tc-F1 at alpha 0.02 (0.809 vs 0.795).
- Best-of-16 temp0.2 (Qwen, 44 unique, baseline-in-candidates): best all-F1 at alpha 0.001 (0.812 vs 0.804); tc-F1 flat (0.805).
- Best-of-32 temp0.2 (Qwen, 19 examples, length_penalty 0.001): best all-F1 at alpha 0.0 (0.806); alpha 0.01 raises all-tc rate (0.842) with flat F1.
- Note: higher alpha can reduce non-typechecked picks; tc-only reranking preserves typecheck rate while improving F1 modestly.

### TinyLlama reference (length-penalty run)

- Run: reference model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, num_candidates 16, temperature 0.2, length_penalty 0.01, start_index 200, 50 requested (44 unique).
- Summary:
  - baseline typecheck 0.750; best_cycle/best_lenpen typecheck 0.773
- Token-level statement similarity:
  - baseline F1 0.771
  - best_cycle/best_lenpen F1 0.794
- Note: cycle scores showed extreme magnitudes (≈1e38) and many non-finite values; ordering still improved F1 on this slice.

### Baseline-in-candidates run (start-index 300)

- Run: num_candidates 16, temperature 0.2, length_penalty 0.01, start_index 300, 50 requested (44 unique); baseline candidate injected into the rerank set.
- Summary:
  - baseline typecheck 0.773; best_cycle typecheck 0.818; best_cycle_tc typecheck 0.864
  - best_lenpen typecheck 0.795; best_lenpen_tc typecheck 0.864
- Token-level statement similarity:
  - baseline F1 0.796
  - best_cycle F1 0.804
  - best_cycle_tc F1 0.805
  - best_lenpen F1 0.797
  - best_lenpen_tc F1 0.799
- Rerank sweep: alpha 0.001 yields best all-F1 (0.812) while preserving all-tc rate (0.818).

### Best-of-32 with length penalty (start-index 350)

- Run: num_candidates 32, temperature 0.2, length_penalty 0.001, start_index 350, 20 requested (19 unique).
- Summary:
  - baseline typecheck 0.895; best_cycle typecheck 0.789; best_cycle_tc typecheck 0.895
  - best_lenpen typecheck 0.789; best_lenpen_tc typecheck 0.895
  - baseline cycle avg -3.196; best_cycle avg -3.086; best_cycle_tc avg -3.092
- Token-level statement similarity:
  - baseline F1 0.811
  - best_cycle F1 0.806
  - best_cycle_tc F1 0.805
  - best_lenpen F1 0.804
  - best_lenpen_tc F1 0.803
- Rerank sweep: alpha 0.0 best all-F1 (0.806); alpha 0.01 improves all-tc rate (0.842) without F1 gains.

### Consensus reranking (candidate agreement)

- New script: `analysis_consensus_rerank.py` (selects candidate with highest average token-F1 agreement to other candidates).
- Baseline-in-candidates slice (44 unique): consensus F1 0.799 vs baseline 0.796; consensus tc-F1 0.793; typecheck rate 0.841.
- Best-of-32 length-penalty slice (19 unique): consensus F1 0.804 vs baseline 0.811; consensus tc-F1 0.805.

### Consensus + cycle sweeps

- New script: `analysis_consensus_cycle_sweep.py` (reranks by `cycle_score - alpha*len + beta*consensus`).
- Baseline-in-candidates slice (44 unique): best all-F1 remains alpha 0.001 beta 0.0 (0.812); consensus does not help here.
- Best-of-16 lenpen001 slice (67 unique): best all-F1 0.819 at beta 2.0–3.0 (vs 0.818 at beta 0.0); tc-F1 0.810–0.812, tc rate 0.881.
- Best-of-32 lenpen001 slice (19 unique): best all-F1 0.812 at beta 3.0 (slightly above baseline 0.811), with all-tc 0.842.

### Qwen 7B GPU logits issue + rescore

- Found Qwen2.5-7B Instruct returns degenerate logits on GPU (bf16 yields uniform -ln|V|, fp16 yields NaNs); cycle scores collapsed to a constant (-11.9375).
- Fix: `load_hf_causal_lm` now sanity-checks logits and reloads the reference model on CPU float32 if degenerate.
- Rescored run: `kimina_bestof_cycle_lenpen001_16_50_start400_rescored.json` (41 unique).
- Token-level statement similarity:
  - baseline F1 0.787
  - best_cycle F1 0.803
  - best_cycle_tc F1 0.811
  - best_lenpen F1 0.798
  - best_lenpen_tc F1 0.806
- Rerank sweep: alpha 0.0 best all-F1 0.803; length penalty hurts; consensus+cycle sweep keeps alpha 0.0 beta 0.0 best.
- Consensus rerank on rescored slice: F1 0.797; tc-F1 0.803; typecheck 0.902.
- Additional GPU checks: Qwen2.5-3B Instruct bf16 on GPU yields NaNs; Qwen2.5-7B still yields zero logits under sdpa/eager/float32, so CPU remains required.

### Qwen 1.5B reference (start-index 450)

- Run: reference model `Qwen/Qwen2.5-1.5B-Instruct`, num_candidates 16, temperature 0.2, length_penalty 0.001, start_index 450, 50 requested (41 unique).
- Summary:
  - baseline typecheck 0.854; best_cycle typecheck 0.878; best_cycle_tc typecheck 1.000
  - best_lenpen typecheck 0.854; best_lenpen_tc typecheck 1.000
  - baseline cycle avg -33.750; best_cycle avg -39.171
- Token-level statement similarity:
  - baseline F1 0.798
  - best_cycle F1 0.795
  - best_cycle_tc F1 0.801
  - best_lenpen F1 0.786
  - best_lenpen_tc F1 0.787
- Rerank sweep: alpha 0.0 best all-F1 (0.795); length penalty hurts.
- Consensus rerank: F1 0.801; tc-F1 0.802; typecheck 0.951.
- Consensus+cycle sweep: best all-F1 0.803 at alpha 0.02 beta 3.0 (all-tc 0.902).

### Qwen 7B rescore on Qwen 1.5B candidates (start-index 450)

- Rescored `kimina_bestof_cycle_lenpen001_16_50_start450_qwen1p5.json` with `Qwen/Qwen2.5-7B-Instruct`.
- Token-level statement similarity:
  - baseline F1 0.798
  - best_cycle F1 0.802
  - best_cycle_tc F1 0.805
  - best_lenpen F1 0.802
  - best_lenpen_tc F1 0.805
- Rerank sweep: best all-F1 0.805 at alpha 0.02 (tc-F1 0.803).
- Consensus rerank: F1 0.809; tc-F1 0.809; typecheck 0.927.
- Consensus+cycle sweep: best all-F1 0.810 at alpha 0.002 beta 3.0 (all-tc 0.878).

### Cycle score vs statement F1 correlation

- New script: `analysis_cycle_f1_correlation.py` (per-example Pearson/Spearman between cycle score and token F1).
- Qwen 1.5B ref (start-index 450): pearson avg -0.082, spearman avg -0.073 (no positive correlation).
- Qwen 7B rescore on same candidates: pearson avg 0.157, spearman avg 0.128 (modest positive correlation).
- Qwen 7B rescore on start-index 400: pearson avg -0.002, spearman avg 0.024 (near zero).

### Qwen 1.5B reference (start-index 500)

- Run: reference model `Qwen/Qwen2.5-1.5B-Instruct`, num_candidates 16, temperature 0.2, length_penalty 0.001, start_index 500, 100 requested (72 unique).
- Summary:
  - baseline typecheck 0.722; best_cycle typecheck 0.736; best_cycle_tc typecheck 0.847
  - best_lenpen typecheck 0.708; best_lenpen_tc typecheck 0.847
- Token-level statement similarity:
  - baseline F1 0.746
  - best_cycle F1 0.765
  - best_cycle_tc F1 0.760
  - best_lenpen F1 0.745
  - best_lenpen_tc F1 0.752
- Rerank sweep: alpha 0.0 best all-F1 0.765; length penalty hurts.
- Consensus rerank: F1 0.765; tc-F1 0.767; typecheck 0.764.
- Consensus+cycle sweep: alpha 0.0 beta 0.5–3.0 keeps all-F1 0.765 while raising all-tc to 0.764.
- Cycle/F1 correlation: pearson avg -0.022, spearman avg -0.016 (no clear correlation).

### Qwen 7B rescore on start-index 500 candidates

- Rescored `kimina_bestof_cycle_lenpen001_16_100_start500_qwen1p5.json` with `Qwen/Qwen2.5-7B-Instruct`.
- Token-level statement similarity:
  - baseline F1 0.746
  - best_cycle F1 0.750
  - best_cycle_tc F1 0.758
  - best_lenpen F1 0.750
  - best_lenpen_tc F1 0.758
- Rerank sweep: best all-F1 0.756 at alpha 0.005.
- Consensus rerank: F1 0.752; tc-F1 0.762; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.768 at alpha 0.005 beta 3.0; tc-F1 0.766.
- Cycle/F1 correlation: pearson avg 0.033, spearman avg 0.080 (weak positive).

### Qwen 1.5B CPU rescore (start-index 500)

- CPU-rescored `kimina_bestof_cycle_lenpen001_16_100_start500_qwen1p5.json` for stable cycle scores.
- Token-level statement similarity:
  - baseline F1 0.746
  - best_cycle F1 0.749
  - best_cycle_tc F1 0.752
  - best_lenpen F1 0.755
- Rerank sweep: alpha 0.005 best all-F1 0.756 (all-tc 0.667).
- Consensus rerank: F1 0.762; tc-F1 0.763; typecheck 0.764.
- Consensus+cycle sweep: best all-F1 0.767 at alpha 0.0 beta 3.0 (all-tc 0.708).
- Learned reranker (CV): F1 0.755; typecheck 0.847.
- Cycle/F1 correlation: pearson avg 0.013, spearman avg 0.045 (weak positive).

### Oracle best-of headroom

- New script: `analysis_oracle_bestof.py` (best achievable token-F1 among candidates).
- start-index 500 (Qwen 1.5B / Qwen 7B rescore): baseline F1 0.746 → oracle 0.803; oracle-tc 0.793; improve rate 0.611.
- start-index 450 (Qwen 1.5B / Qwen 7B rescore): baseline F1 0.798 → oracle 0.842; oracle-tc 0.830; improve rate 0.659.
- start-index 400 (Qwen 7B rescore): baseline F1 0.787 → oracle 0.850; oracle-tc 0.844; improve rate 0.610.

### Learned reranker (linear features + CV)

- New script: `analysis_learned_reranker.py` (cycle/length/consensus/typecheck z-scores with 5-fold CV).
- start-index 500 rescored (Qwen 7B): baseline F1 0.746 → learned 0.757; typecheck 0.847.
- start-index 500 (Qwen 1.5B): baseline F1 0.746 → learned 0.758; typecheck 0.847.
- start-index 450 rescored (Qwen 7B): baseline F1 0.798 → learned 0.811; typecheck 1.000.
- start-index 400 rescored (Qwen 7B): baseline F1 0.787 → learned 0.803; typecheck 0.927.
- Temp‑0.8 merged 128-candidate slices (start-index 1580–2100, 163 records): baseline F1 0.766 → learned 0.842 (tc-F1 0.843; typecheck 0.939); learned beats consensus 0.834.
- Temp‑0.8 merged 128-candidate slices (start-index 1580–2100): no-cycle 0.842 (slightly lower); +baseline-sim feature 0.843; GBRT drops to 0.826; no-typecheck 0.836; no-consensus collapses to 0.767.

### Learned reranker (train/test transfer)

- Train on start-index 500 rescore (Qwen 7B), test on start-index 450 rescore: baseline F1 0.798 → learned 0.811; typecheck 1.000.
- Train on start-index 500 rescore (Qwen 7B), test on start-index 400 rescore: baseline F1 0.787 → learned 0.810; typecheck 0.927.
- Train on start-index 500 (Qwen 1.5B), test on start-index 450 (Qwen 1.5B): baseline F1 0.798 → learned 0.806; typecheck 1.000.
- Train on temp‑0.8 merged 128-candidate slices (start-index 1580–2080), test on start-index 2100: baseline F1 0.801 → learned 0.810; typecheck 0.947.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2100, 513 records): baseline F1 0.778 → learned 0.844; typecheck 0.957.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature: baseline F1 0.778 → learned 0.844; typecheck 0.959.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2120, 532 records): baseline F1 0.779 → learned 0.843; typecheck 0.957.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2120): baseline F1 0.779 → learned 0.844; typecheck 0.959.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2160, 571 records): baseline F1 0.777 → learned 0.841; typecheck 0.956.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2160): baseline F1 0.777 → learned 0.842; typecheck 0.956.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2180, 591 records): baseline F1 0.776 → learned 0.843; typecheck 0.958.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2180): baseline F1 0.776 → learned 0.843; typecheck 0.958.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2200, 611 records): baseline F1 0.776 → learned 0.842; typecheck 0.956.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2200): baseline F1 0.776 → learned 0.843; typecheck 0.956.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2220, 630 records): baseline F1 0.776 → learned 0.840; typecheck 0.956.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2220): baseline F1 0.776 → learned 0.841; typecheck 0.957.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2240, 650 records): baseline F1 0.778 → learned 0.841; typecheck 0.957.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2240): baseline F1 0.778 → learned 0.843; typecheck 0.957.
- Leave-one-slice-out over temp‑0.8 slices (start-index 1580–2260, 668 records): baseline F1 0.777 → learned 0.840; typecheck 0.958.
- Leave-one-slice-out over temp‑0.8 slices with baseline feature (start-index 1580–2260): baseline F1 0.777 → learned 0.841; typecheck 0.958.
- Learned vs consensus across slices (1580–2100): learned beats all‑consensus in 20/27 slices, with avg +0.0054 F1; best gain +0.0245 (start1680), worst ‑0.0130 (start1860).
- Learned vs consensus across slices (1580–2120): learned beats all‑consensus in 20/28 slices, with avg +0.0053 F1; best gain +0.0245 (start1680), worst ‑0.0130 (start1860).
- Learned vs consensus across slices (1580–2160): learned beats all‑consensus in 23/30 slices, with avg +0.0055 F1; best gain +0.0245 (start1680), worst ‑0.0130 (start1860).
- Learned vs consensus across slices (1580–2180): learned beats all‑consensus in 24/31 slices, with avg +0.0059 F1; best gain +0.0245 (start1680), worst ‑0.0130 (start1860).
- Learned vs consensus across slices (1580–2200): learned beats all‑consensus in 23/32 slices, with avg +0.0048 F1; best gain +0.0245 (start1680), worst ‑0.0130 (start1860).
- Learned vs consensus across slices (1580–2220): learned beats all‑consensus in 24/33 slices, with avg +0.0047 F1; best gain +0.0245 (start1680), worst ‑0.0109 (start1760).
- Learned vs consensus across slices (1580–2240): learned beats all‑consensus in 23/34 slices, with avg +0.0044 F1; best gain +0.0245 (start1680), worst ‑0.0138 (start2060).
- Learned vs consensus across slices (1580–2260): learned beats all‑consensus in 24/35 slices, with avg +0.0043 F1; best gain +0.0245 (start1680), worst ‑0.0138 (start2060).
- Train on temp‑0.8 start-index 740, test on temp‑0.6 start-index 700 (64 candidates): baseline F1 0.789 → learned 0.855; typecheck 1.000.
- Train on temp‑0.6 start-index 700, test on temp‑0.8 start-index 740 (64 candidates): baseline F1 0.746 → learned 0.804; typecheck 0.889.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 760: baseline F1 0.702 → learned 0.794; typecheck 0.941.
- Train on temp‑0.8 start-index 760, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.832; typecheck 0.944.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 780: baseline F1 0.759 → learned 0.825; typecheck 1.000.
- Train on temp‑0.8 start-index 780, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.807; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 800: baseline F1 0.772 → learned 0.785; typecheck 0.944.
- Train on temp‑0.8 start-index 800, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 820: baseline F1 0.815 → learned 0.850; typecheck 0.947.
- Train on temp‑0.8 start-index 820, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.804; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 840: baseline F1 0.822 → learned 0.803; typecheck 1.000.
- Train on temp‑0.8 start-index 840, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.8 merged (start-index 740–820), test on temp‑0.8 start-index 840: baseline F1 0.822 → learned 0.813; typecheck 1.000.
- Train on temp‑0.8 merged (start-index 740–840), test on temp‑0.6 start-index 700: baseline F1 0.789 → learned 0.856; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 860: baseline F1 0.817 → learned 0.854; typecheck 1.000.
- Train on temp‑0.8 start-index 860, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 880: baseline F1 0.733 → learned 0.802; typecheck 0.900.
- Train on temp‑0.8 start-index 880, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.808; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 880: baseline F1 0.733 → learned 0.815; typecheck 0.900.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 900: baseline F1 0.702 → learned 0.827; typecheck 0.947.
- Train on temp‑0.8 start-index 900, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.803; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 900: baseline F1 0.702 → learned 0.823; typecheck 0.947.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 920: baseline F1 0.785 → learned 0.871; typecheck 1.000.
- Train on temp‑0.8 start-index 920, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.822; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 920: baseline F1 0.785 → learned 0.869; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 940: baseline F1 0.779 → learned 0.853; typecheck 1.000.
- Train on temp‑0.8 start-index 940, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 0.944.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 940: baseline F1 0.779 → learned 0.853; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 960: baseline F1 0.804 → learned 0.832; typecheck 1.000.
- Train on temp‑0.8 start-index 960, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 960: baseline F1 0.804 → learned 0.833; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 980: baseline F1 0.753 → learned 0.823; typecheck 1.000.
- Train on temp‑0.8 start-index 980, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.808; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 980: baseline F1 0.753 → learned 0.823; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1000: baseline F1 0.831 → learned 0.889; typecheck 1.000.
- Train on temp‑0.8 start-index 1000, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.824; typecheck 0.778.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1000: baseline F1 0.831 → learned 0.889; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 860: baseline F1 0.817 → learned 0.855; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 840: baseline F1 0.822 → learned 0.802; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1020: baseline F1 0.790 → learned 0.850; typecheck 0.941.
- Train on temp‑0.8 start-index 1020, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.826; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1020: baseline F1 0.790 → learned 0.877; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1040: baseline F1 0.760 → learned 0.769; typecheck 1.000.
- Train on temp‑0.8 start-index 1040, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 0.944.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1040: baseline F1 0.760 → learned 0.771; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1060: baseline F1 0.788 → learned 0.823; typecheck 1.000.
- Train on temp‑0.8 start-index 1060, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.814; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1060: baseline F1 0.788 → learned 0.823; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1080: baseline F1 0.784 → learned 0.815; typecheck 1.000.
- Train on temp‑0.8 start-index 1080, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1080: baseline F1 0.784 → learned 0.809; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1100: baseline F1 0.766 → learned 0.800; typecheck 1.000.
- Train on temp‑0.8 start-index 1100, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.803; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1100: baseline F1 0.766 → learned 0.801; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1120: baseline F1 0.803 → learned 0.850; typecheck 1.000.
- Train on temp‑0.8 start-index 1120, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1120: baseline F1 0.803 → learned 0.854; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1140: baseline F1 0.748 → learned 0.792; typecheck 0.900.
- Train on temp‑0.8 start-index 1140, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1140: baseline F1 0.748 → learned 0.797; typecheck 0.900.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1160: baseline F1 0.792 → learned 0.827; typecheck 1.000.
- Train on temp‑0.8 start-index 1160, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.796; typecheck 0.778.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1160: baseline F1 0.792 → learned 0.826; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1180: baseline F1 0.737 → learned 0.801; typecheck 1.000.
- Train on temp‑0.8 start-index 1180, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.814; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1180: baseline F1 0.737 → learned 0.801; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1200: baseline F1 0.824 → learned 0.854; typecheck 1.000.
- Train on temp‑0.8 start-index 1200, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.830; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1200: baseline F1 0.824 → learned 0.856; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1220: baseline F1 0.752 → learned 0.817; typecheck 0.947.
- Train on temp‑0.8 start-index 1220, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1220: baseline F1 0.752 → learned 0.822; typecheck 0.947.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1240: baseline F1 0.758 → learned 0.814; typecheck 1.000.
- Train on temp‑0.8 start-index 1240, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.810; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1240: baseline F1 0.758 → learned 0.811; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1260: baseline F1 0.728 → learned 0.812; typecheck 0.895.
- Train on temp‑0.8 start-index 1260, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1260: baseline F1 0.728 → learned 0.818; typecheck 0.947.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1280: baseline F1 0.746 → learned 0.800; typecheck 0.947.
- Train on temp‑0.8 start-index 1280, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.825; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1280: baseline F1 0.746 → learned 0.811; typecheck 0.947.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1300: baseline F1 0.792 → learned 0.825; typecheck 1.000.
- Train on temp‑0.8 start-index 1300, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.809; typecheck 0.778.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1300: baseline F1 0.792 → learned 0.820; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1320: baseline F1 0.773 → learned 0.837; typecheck 1.000.
- Train on temp‑0.8 start-index 1320, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.808; typecheck 0.889.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1320: baseline F1 0.773 → learned 0.830; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1340: baseline F1 0.835 → learned 0.846; typecheck 1.000.
- Train on temp‑0.8 start-index 1340, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.802; typecheck 0.889.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1340: baseline F1 0.835 → learned 0.846; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1360: baseline F1 0.731 → learned 0.784; typecheck 1.000.
- Train on temp‑0.8 start-index 1360, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.799; typecheck 0.889.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1360: baseline F1 0.731 → learned 0.777; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1380: baseline F1 0.768 → learned 0.844; typecheck 1.000.
- Train on temp‑0.8 start-index 1380, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.812; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1380: baseline F1 0.768 → learned 0.825; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1400: baseline F1 0.832 → learned 0.870; typecheck 1.000.
- Train on temp‑0.8 start-index 1400, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.804; typecheck 0.889.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1400: baseline F1 0.832 → learned 0.868; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1420: baseline F1 0.816 → learned 0.876; typecheck 1.000.
- Train on temp‑0.8 start-index 1420, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.816; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1420: baseline F1 0.816 → learned 0.872; typecheck 1.000.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1440: baseline F1 0.788 → learned 0.818; typecheck 0.889.
- Train on temp‑0.8 start-index 1440, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.811; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1440: baseline F1 0.788 → learned 0.811; typecheck 0.944.
- Train on temp‑0.8 start-index 740, test on temp‑0.8 start-index 1460: baseline F1 0.755 → learned 0.817; typecheck 1.000.
- Train on temp‑0.8 start-index 1460, test on temp‑0.8 start-index 740: baseline F1 0.746 → learned 0.826; typecheck 1.000.
- Train on temp‑0.6 merged (start-index 600–700), test on temp‑0.8 start-index 1460: baseline F1 0.755 → learned 0.795; typecheck 1.000.
- Train on temp‑0.8 merged (start-index 740–1460), test on temp‑0.8 start-index 1940 (128 candidates): baseline F1 0.785 → learned 0.833; typecheck 0.950.
- Train on temp‑0.8 merged (start-index 740–1460), test on temp‑0.8 start-index 1960 (128 candidates): baseline F1 0.764 → learned 0.845; typecheck 0.947.
- Train on temp‑0.8 merged (start-index 740–1460), test on temp‑0.8 start-index 1980 (128 candidates): baseline F1 0.749 → learned 0.804; typecheck 0.950.
- Train on temp‑0.8 merged (start-index 740–1460), test on temp‑0.8 start-index 2000 (128 candidates): baseline F1 0.761 → learned 0.823; typecheck 0.947.

### High-temperature best-of (temp 0.6, Qwen 1.5B)

- Run: num_candidates 16, temperature 0.6, length_penalty 0.001, start_index 600, 50 requested (44 unique).
- Initial run produced many non-finite cycle scores; rescoring on CPU float32 fixed this.
- CPU-rescored summary (cycle-based):
  - baseline F1 0.755
  - best_cycle F1 0.774
  - best_cycle_tc F1 0.762
  - best_lenpen F1 0.760
- Consensus rerank (unchanged by rescore): F1 0.778; tc-F1 0.784; typecheck 0.864.
- Consensus+cycle sweep (CPU-rescored): best all-F1 0.788 at alpha 0.0 beta 2.0 (all-tc 0.795; tc-F1 0.792).
- Learned reranker (CV on CPU-rescored): F1 0.783; typecheck 0.955.
- Oracle headroom: baseline F1 0.755 → oracle 0.875; improve rate 0.750.
- Cycle/F1 correlation (CPU-rescored): pearson avg 0.088, spearman avg 0.061 (weak positive).
- Added `--force-cpu` to `analysis_rescore_cycle.py` (and `force_cpu` in `load_hf_causal_lm`) to handle GPU NaN cases.
- Learned reranker transfer: train on start-index 500 CPU-rescored (Qwen 1.5B), test on temp 0.6 CPU-rescored; F1 0.775 vs baseline 0.755 (typecheck 0.955).

### Typecheck signal strength

- New script: `analysis_typecheck_signal.py` (compares F1 for typechecked vs non-typechecked candidates).
- start-index 500 CPU-rescored (Qwen 1.5B): tc candidate F1 0.814 vs non-tc 0.671; max tc F1 0.838 vs max non-tc 0.680.
- temp 0.6 CPU-rescored: tc candidate F1 0.775 vs non-tc 0.673; max tc F1 0.870 vs max non-tc 0.769.

### High-temp + more candidates (temp 0.6, 32 candidates)

- Run: num_candidates 32, temperature 0.6, length_penalty 0.001, start_index 650, 20 requested (20 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.767
  - best_cycle F1 0.727 (tc-F1 0.770)
- Consensus rerank: F1 0.808; tc-F1 0.806; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.787 at alpha 0.0 beta 3.0; tc-F1 0.808.
- Learned reranker (CV): F1 0.803; tc-F1 0.814; typecheck 0.900.
- Learned reranker transfer (train start-index 500 CPU-rescored): F1 0.788; typecheck 1.000.
- Oracle headroom: baseline F1 0.767 → oracle 0.881; oracle-tc 0.855.
- Cycle/F1 correlation: pearson avg 0.137, spearman avg 0.089.

### High-temp + 64 candidates (temp 0.6, 64 candidates)

- Run: num_candidates 64, temperature 0.6, length_penalty 0.001, start_index 700, 20 requested (19 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.789
  - best_cycle F1 0.779
  - best_cycle_tc F1 0.799
  - best_lenpen_tc F1 0.806
- Rerank sweep: best all-F1 0.786 at alpha 0.001; tc-F1 0.806 with tc rate 1.000.
- Consensus rerank: F1 0.853; tc-F1 0.838; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.847 at alpha 0.0 beta 2.0; tc-F1 0.850.
- Learned reranker (CV): F1 0.831; tc-F1 0.848; typecheck 0.842.
- Oracle headroom: baseline F1 0.789 → oracle 0.931 (all/tc); improve rate 0.895.
- Cycle/F1 correlation: pearson avg 0.184, spearman avg 0.118.

### High-temp + 64 candidates (temp 0.6, start-index 720)

- Run: num_candidates 64, temperature 0.6, length_penalty 0.001, start_index 720, 20 requested (17 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.704
  - best_cycle F1 0.777 (tc-F1 0.788)
  - best_lenpen F1 0.771 (tc-F1 0.782)
- Rerank sweep: best all-F1 0.777 at alpha 0.0 (tc-F1 0.788; tc-rate 1.000).
- Consensus rerank: F1 0.748; tc-F1 0.773; typecheck 0.824.
- Consensus+cycle sweep: best all-F1 0.777 at alpha 0.0 beta 0.0 (tc-F1 0.788).
- Learned reranker (CV): F1 0.753; tc-F1 0.776; typecheck 0.824.
- Oracle headroom: baseline F1 0.704 → oracle 0.867; oracle-tc 0.865.
- Cycle/F1 correlation: pearson avg 0.315, spearman avg 0.239.
- Typecheck signal: tc candidate F1 0.724 vs non-tc 0.714; max tc F1 0.865 vs max non-tc 0.650.
- Consensus k-sweep (all candidates): k=4 F1 0.735, k=8 F1 0.754, k=16 F1 0.766, k=32 F1 0.737, k=64 F1 0.690.
- Consensus all vs tc-only: all-consensus F1 0.748 vs tc-only 0.773; tc-only wins 29.4%.

### High-temp + 64 candidates (temp 0.8, start-index 740)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 740, 20 requested (18 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.746
  - best_cycle F1 0.779 (tc-F1 0.762)
  - best_lenpen F1 0.767 (tc-F1 0.754)
- Rerank sweep: best all-F1 0.779 at alpha 0.0 (tc-F1 0.762; tc-rate 1.000).
- Consensus rerank: F1 0.797; tc-F1 0.800; typecheck 0.833.
- Consensus+cycle sweep: best all-F1 0.804 at alpha 0.002 beta 3.0; tc-F1 0.809.
- Learned reranker (CV): F1 0.810; tc-F1 0.810; typecheck 1.000.
- Oracle headroom: baseline F1 0.746 → oracle 0.936; oracle-tc 0.926.
- Cycle/F1 correlation: pearson avg 0.283, spearman avg 0.209.
- Typecheck signal: tc candidate F1 0.775 vs non-tc 0.667; max tc F1 0.926 vs max non-tc 0.778.
- Consensus k-sweep (all candidates): k=4 F1 0.781, k=8 F1 0.795, k=16 F1 0.814, k=32 F1 0.812, k=64 F1 0.811.
- Consensus all vs tc-only: all-consensus F1 0.797 vs tc-only 0.800; tc-only wins 16.7%.

### High-temp + 64 candidates (temp 0.8, start-index 760)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 760, 20 requested (17 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.702
  - best_cycle F1 0.713 (tc-F1 0.690)
  - best_lenpen F1 0.719 (tc-F1 0.696)
- Rerank sweep: best all-F1 0.719 at alpha 0.001 (tc-F1 0.696; tc-rate 1.000).
- Consensus rerank: F1 0.794; tc-F1 0.766; typecheck 0.941.
- Consensus+cycle sweep: best all-F1 0.794 at alpha 0.0 beta 2.0 (tc-F1 0.754).
- Learned reranker (CV): F1 0.798; tc-F1 0.762; typecheck 0.941.
- Oracle headroom: baseline F1 0.702 → oracle 0.924; oracle-tc 0.869.
- Cycle/F1 correlation: pearson avg 0.333, spearman avg 0.115.
- Typecheck signal: tc candidate F1 0.756 vs non-tc 0.699; max tc F1 0.869 vs max non-tc 0.823.
- Consensus k-sweep (all candidates): k=4 F1 0.769, k=8 F1 0.785, k=16 F1 0.795, k=32 F1 0.794, k=64 F1 0.837.
- Consensus all vs tc-only: all-consensus F1 0.794 vs tc-only 0.766; tc-only wins 11.8%.

### High-temp + 64 candidates (temp 0.8, start-index 780)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 780, 20 requested (20 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.759
  - best_cycle F1 0.780 (tc-F1 0.759)
  - best_lenpen F1 0.779 (tc-F1 0.763)
- Rerank sweep: best all-F1 0.780 at alpha 0.0 (tc-F1 0.759; tc-rate 1.000).
- Consensus rerank: F1 0.826; tc-F1 0.814; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.821 at alpha 0.0 beta 3.0 (tc-F1 0.816).
- Learned reranker (CV): F1 0.823; tc-F1 0.823; typecheck 1.000.
- Oracle headroom: baseline F1 0.759 → oracle 0.926; oracle-tc 0.905.
- Cycle/F1 correlation: pearson avg 0.194, spearman avg 0.105.
- Typecheck signal: tc candidate F1 0.802 vs non-tc 0.715; max tc F1 0.905 vs max non-tc 0.863.
- Consensus k-sweep (all candidates): k=4 F1 0.809, k=8 F1 0.821, k=16 F1 0.827, k=32 F1 0.818, k=64 F1 0.859.
- Consensus all vs tc-only: all-consensus F1 0.826 vs tc-only 0.814; tc-only wins 15.0%.

### High-temp + 64 candidates (temp 0.8, start-index 800)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 800, 20 requested (18 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.772
  - best_cycle F1 0.736 (tc-F1 0.735)
  - best_lenpen F1 0.721 (tc-F1 0.737)
- Rerank sweep: best all-F1 0.736 at alpha 0.0 (tc-F1 0.735; tc-rate 0.944).
- Consensus rerank: F1 0.793; tc-F1 0.794; typecheck 0.944.
- Consensus+cycle sweep: best all-F1 0.776 at alpha 0.002 beta 2.0 (tc-F1 0.778).
- Learned reranker (CV): F1 0.784; tc-F1 0.784; typecheck 0.944.
- Oracle headroom: baseline F1 0.772 → oracle 0.914; oracle-tc 0.904.
- Cycle/F1 correlation: pearson avg 0.299, spearman avg 0.122.
- Typecheck signal: tc candidate F1 0.749 vs non-tc 0.644; max tc F1 0.918 vs max non-tc 0.789.
- Consensus k-sweep (all candidates): k=4 F1 0.766, k=8 F1 0.778, k=16 F1 0.788, k=32 F1 0.777.
- Consensus all vs tc-only: all-consensus F1 0.793 vs tc-only 0.794; tc-only wins 11.8%.

### High-temp + 64 candidates (temp 0.8, start-index 820)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 820, 20 requested (19 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.815
  - best_cycle F1 0.730 (tc-F1 0.752)
  - best_lenpen F1 0.759 (tc-F1 0.764)
- Rerank sweep: best all-F1 0.759 at alpha 0.001 (tc-F1 0.764; tc-rate 1.000).
- Consensus rerank: F1 0.835; tc-F1 0.851; typecheck 0.684.
- Consensus+cycle sweep: best all-F1 0.841 at alpha 0.005 beta 3.0 (tc-F1 0.845).
- Learned reranker (CV): F1 0.848; tc-F1 0.840; typecheck 0.947.
- Oracle headroom: baseline F1 0.815 → oracle 0.925; oracle-tc 0.899.
- Cycle/F1 correlation: pearson avg 0.315, spearman avg 0.213.
- Typecheck signal: tc candidate F1 0.816 vs non-tc 0.708; max tc F1 0.899 vs max non-tc 0.853.
- Consensus k-sweep (all candidates): k=4 F1 0.812, k=8 F1 0.827, k=16 F1 0.836, k=32 F1 0.829, k=64 F1 0.825.
- Consensus all vs tc-only: all-consensus F1 0.835 vs tc-only 0.851; tc-only wins 31.6%.

### High-temp + 64 candidates (temp 0.8, start-index 840)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 840, 20 requested (19 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.822
  - best_cycle F1 0.755 (tc-F1 0.779)
  - best_lenpen F1 0.761 (tc-F1 0.785)
- Rerank sweep: best all-F1 0.777 at alpha 0.002 (tc-F1 0.785; tc-rate 1.000).
- Consensus rerank: F1 0.824; tc-F1 0.810; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.801 at alpha 0.005 beta 2.0 (tc-F1 0.782).
- Learned reranker (CV): F1 0.803; tc-F1 0.803; typecheck 1.000.
- Oracle headroom: baseline F1 0.822 → oracle 0.946; oracle-tc 0.931.
- Cycle/F1 correlation: pearson avg 0.323, spearman avg 0.125.
- Typecheck signal: tc candidate F1 0.771 vs non-tc 0.708; max tc F1 0.931 vs max non-tc 0.805.
- Consensus k-sweep (all candidates): k=4 F1 0.797, k=8 F1 0.808, k=16 F1 0.811, k=32 F1 0.805, k=64 F1 0.861.
- Consensus all vs tc-only: all-consensus F1 0.824 vs tc-only 0.810; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 860)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 860, 20 requested (20 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.817
  - best_cycle F1 0.753 (tc-F1 0.770)
  - best_lenpen F1 0.752 (tc-F1 0.770)
- Rerank sweep: best all-F1 0.753 at alpha 0.0 (tc-F1 0.770; tc-rate 1.000).
- Consensus rerank: F1 0.841; tc-F1 0.855; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.829 at alpha 0.002 beta 3.0 (tc-F1 0.841).
- Learned reranker (CV): F1 0.851; tc-F1 0.851; typecheck 1.000.
- Oracle headroom: baseline F1 0.817 → oracle 0.919; oracle-tc 0.916.
- Cycle/F1 correlation: pearson avg 0.289, spearman avg 0.142.
- Typecheck signal: tc candidate F1 0.790 vs non-tc 0.729; max tc F1 0.916 vs max non-tc 0.786.
- Consensus k-sweep (all candidates): k=4 F1 0.818, k=8 F1 0.825, k=16 F1 0.836, k=32 F1 0.829, k=64 F1 0.772.
- Consensus all vs tc-only: all-consensus F1 0.841 vs tc-only 0.855; tc-only wins 30.0%.

### High-temp + 64 candidates (temp 0.8, start-index 880)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 880, 20 requested (20 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.733
  - best_cycle F1 0.742 (tc-F1 0.775)
  - best_lenpen F1 0.735 (tc-F1 0.764)
- Rerank sweep: best all-F1 0.742 at alpha 0.0 (tc-F1 0.775; tc-rate 0.900).
- Consensus rerank: F1 0.794; tc-F1 0.798; typecheck 0.600.
- Consensus+cycle sweep: best all-F1 0.805 at alpha 0.0 beta 3.0 (tc-F1 0.807).
- Learned reranker (CV): F1 0.794; tc-F1 0.803; typecheck 0.850.
- Oracle headroom: baseline F1 0.733 → oracle 0.918; oracle-tc 0.897.
- Cycle/F1 correlation: pearson avg 0.262, spearman avg 0.219.
- Typecheck signal: tc candidate F1 0.764 vs non-tc 0.686; max tc F1 0.910 vs max non-tc 0.793.
- Consensus k-sweep (all candidates): k=4 F1 0.772, k=8 F1 0.780, k=16 F1 0.788, k=32 F1 0.779, k=64 F1 0.850.
- Consensus all vs tc-only: all-consensus F1 0.794 vs tc-only 0.787; tc-only wins 11.1%.

### High-temp + 64 candidates (temp 0.8, start-index 900)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 900, 20 requested (19 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.702
  - best_cycle F1 0.726 (tc-F1 0.787)
  - best_lenpen F1 0.725 (tc-F1 0.740)
- Rerank sweep: best all-F1 0.726 at alpha 0.0 (tc-F1 0.787; tc-rate 1.000).
- Consensus rerank: F1 0.788; tc-F1 0.822; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.797 at alpha 0.005 beta 3.0 (tc-F1 0.829).
- Learned reranker (CV): F1 0.816; tc-F1 0.825; typecheck 0.947.
- Oracle headroom: baseline F1 0.702 → oracle 0.921; oracle-tc 0.911.
- Cycle/F1 correlation: pearson avg 0.345, spearman avg 0.201.
- Typecheck signal: tc candidate F1 0.764 vs non-tc 0.617; max tc F1 0.911 vs max non-tc 0.775.
- Consensus k-sweep (all candidates): k=4 F1 0.778, k=8 F1 0.780, k=16 F1 0.786, k=32 F1 0.773, k=64 F1 0.878.
- Consensus all vs tc-only: all-consensus F1 0.788 vs tc-only 0.822; tc-only wins 21.1%.

### High-temp + 64 candidates (temp 0.8, start-index 920)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 920, 20 requested (20 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.785
  - best_cycle F1 0.779 (tc-F1 0.792)
  - best_lenpen F1 0.724 (tc-F1 0.790)
- Rerank sweep: best all-F1 0.779 at alpha 0.0 (tc-F1 0.792; tc-rate 1.000).
- Consensus rerank: F1 0.870; tc-F1 0.870; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.870 at alpha 0.0 beta 3.0 (tc-F1 0.871).
- Learned reranker (CV): F1 0.868; tc-F1 0.863; typecheck 0.950.
- Oracle headroom: baseline F1 0.785 → oracle 0.928; oracle-tc 0.921.
- Cycle/F1 correlation: pearson avg 0.329, spearman avg 0.248.
- Typecheck signal: tc candidate F1 0.797 vs non-tc 0.726; max tc F1 0.921 vs max non-tc 0.861.
- Consensus k-sweep (all candidates): k=4 F1 0.836, k=8 F1 0.852, k=16 F1 0.865, k=32 F1 0.861, k=64 F1 0.833.
- Consensus all vs tc-only: all-consensus F1 0.870 vs tc-only 0.870; tc-only wins 10.0%.

### High-temp + 64 candidates (temp 0.8, start-index 940)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 940, 20 requested (18 unique), CPU-rescored.
- Token-level statement similarity:
  - baseline F1 0.779
  - best_cycle F1 0.775 (tc-F1 0.793)
  - best_lenpen F1 0.775 (tc-F1 0.793)
- Rerank sweep: best all-F1 0.787 at alpha 0.002 (tc-F1 0.807; tc-rate 1.000).
- Consensus rerank: F1 0.841; tc-F1 0.841; typecheck 0.944.
- Consensus+cycle sweep: best all-F1 0.828 at alpha 0.001 beta 2.0 (tc-F1 0.836).
- Learned reranker (CV): F1 0.837; tc-F1 0.837; typecheck 1.000.
- Oracle headroom: baseline F1 0.779 → oracle 0.913; oracle-tc 0.911.
- Cycle/F1 correlation: pearson avg 0.334, spearman avg 0.206.
- Typecheck signal: tc candidate F1 0.786 vs non-tc 0.722; max tc F1 0.911 vs max non-tc 0.808.
- Consensus k-sweep (all candidates): k=4 F1 0.816, k=8 F1 0.828, k=16 F1 0.836, k=32 F1 0.833, k=64 F1 0.892.
- Consensus all vs tc-only: all-consensus F1 0.841 vs tc-only 0.841; tc-only wins 5.6%.

### High-temp + 64 candidates (temp 0.8, start-index 960)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 960, 20 requested (17 unique), CPU-rescored using local Qwen2.5-1.5B snapshot path to avoid HF hang.
- Token-level statement similarity:
  - baseline F1 0.804
  - best_cycle F1 0.686 (tc-F1 0.730)
  - best_lenpen F1 0.683 (tc-F1 0.710)
- Rerank sweep: best all-F1 0.686 at alpha 0.0 (tc-F1 0.730; tc-rate 1.000).
- Consensus rerank: F1 0.837; tc-F1 0.838; typecheck 0.941.
- Consensus+cycle sweep: best all-F1 0.844 at alpha 0.005 beta 3.0 (tc-F1 0.850).
- Learned reranker (CV): F1 0.833; tc-F1 0.833; typecheck 1.000.
- Oracle headroom: baseline F1 0.804 → oracle 0.936; oracle-tc 0.926.
- Cycle/F1 correlation: pearson avg 0.134, spearman avg 0.025.
- Typecheck signal: tc candidate F1 0.791 vs non-tc 0.674; max tc F1 0.926 vs max non-tc 0.870.
- Consensus k-sweep (all candidates): k=4 F1 0.809, k=8 F1 0.825, k=16 F1 0.832, k=32 F1 0.825, k=64 F1 0.805.
- Consensus all vs tc-only: all-consensus F1 0.837 vs tc-only 0.838; tc-only wins 11.8%.

### High-temp + 64 candidates (temp 0.8, start-index 980)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 980, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.753
  - best_cycle F1 0.765 (tc-F1 0.762)
  - best_lenpen F1 0.758 (tc-F1 0.765)
- Rerank sweep: best all-F1 0.772 at alpha 0.005 (tc-F1 0.772; tc-rate 1.000).
- Consensus rerank: F1 0.810; tc-F1 0.800; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.808 at alpha 0.002 beta 3.0 (tc-F1 0.805).
- Learned reranker (CV): F1 0.829; tc-F1 0.825; typecheck 0.950.
- Oracle headroom: baseline F1 0.753 → oracle 0.928; oracle-tc 0.920.
- Cycle/F1 correlation: pearson avg 0.356, spearman avg 0.241.
- Typecheck signal: tc candidate F1 0.738 vs non-tc 0.682; max tc F1 0.920 vs max non-tc 0.848.
- Consensus k-sweep (all candidates): k=4 F1 0.759, k=8 F1 0.788, k=16 F1 0.802, k=32 F1 0.801, k=64 F1 0.933.
- Consensus all vs tc-only: all-consensus F1 0.810 vs tc-only 0.800; tc-only wins 20.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1000)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1000, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.831
  - best_cycle F1 0.824 (tc-F1 0.835)
  - best_lenpen F1 0.829 (tc-F1 0.830)
- Rerank sweep: best all-F1 0.829 at alpha 0.001 (tc-F1 0.830; tc-rate 1.000).
- Consensus rerank: F1 0.861; tc-F1 0.889; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.862 at alpha 0.0 beta 3.0 (tc-F1 0.886).
- Learned reranker (CV): F1 0.872; tc-F1 0.878; typecheck 0.842.
- Oracle headroom: baseline F1 0.831 → oracle 0.955; oracle-tc 0.953.
- Cycle/F1 correlation: pearson avg 0.419, spearman avg 0.255.
- Typecheck signal: tc candidate F1 0.817 vs non-tc 0.782; max tc F1 0.953 vs max non-tc 0.862.
- Consensus k-sweep (all candidates): k=4 F1 0.849, k=8 F1 0.852, k=16 F1 0.870, k=32 F1 0.870, k=64 F1 0.860.
- Consensus all vs tc-only: all-consensus F1 0.861 vs tc-only 0.889; tc-only wins 31.6%.

### High-temp + 64 candidates (temp 0.8, start-index 1020)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1020, 20 requested (17 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.790
  - best_cycle F1 0.799 (tc-F1 0.766)
  - best_lenpen F1 0.798 (tc-F1 0.775)
- Rerank sweep: best all-F1 0.799 at alpha 0.0 (tc-F1 0.766; tc-rate 1.000).
- Consensus rerank: F1 0.846; tc-F1 0.881; typecheck 0.647.
- Consensus+cycle sweep: best all-F1 0.843 at alpha 0.001 beta 2.0 (tc-F1 0.875).
- Learned reranker (CV): F1 0.844; tc-F1 0.866; typecheck 0.882.
- Oracle headroom: baseline F1 0.790 → oracle 0.959; oracle-tc 0.941.
- Cycle/F1 correlation: pearson avg 0.341, spearman avg 0.258.
- Typecheck signal: tc candidate F1 0.838 vs non-tc 0.729; max tc F1 0.941 vs max non-tc 0.909.
- Consensus k-sweep (all candidates): k=4 F1 0.829, k=8 F1 0.844, k=16 F1 0.847, k=32 F1 0.846, k=64 F1 0.822.
- Consensus all vs tc-only: all-consensus F1 0.846 vs tc-only 0.881; tc-only wins 35.3%.

### High-temp + 64 candidates (temp 0.8, start-index 1040)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1040, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.760
  - best_cycle F1 0.705 (tc-F1 0.717)
  - best_lenpen F1 0.702 (tc-F1 0.717)
- Rerank sweep: best all-F1 0.705 at alpha 0.002 (tc-F1 0.677; tc-rate 1.000).
- Consensus rerank: F1 0.768; tc-F1 0.746; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.762 at alpha 0.0 beta 2.0 (tc-F1 0.768).
- Learned reranker (CV): F1 0.770; tc-F1 0.774; typecheck 0.800.
- Oracle headroom: baseline F1 0.760 → oracle 0.924; oracle-tc 0.900.
- Cycle/F1 correlation: pearson avg 0.203, spearman avg 0.119.
- Typecheck signal: tc candidate F1 0.726 vs non-tc 0.697; max tc F1 0.900 vs max non-tc 0.836.
- Consensus k-sweep (all candidates): k=4 F1 0.752, k=8 F1 0.756, k=16 F1 0.763, k=32 F1 0.759, k=64 F1 0.725.
- Consensus all vs tc-only: all-consensus F1 0.768 vs tc-only 0.746; tc-only wins 10.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1060)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1060, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.788
  - best_cycle F1 0.749 (tc-F1 0.773)
  - best_lenpen F1 0.747 (tc-F1 0.771)
- Rerank sweep: best all-F1 0.749 at alpha 0.0 (tc-F1 0.773; tc-rate 1.000).
- Consensus rerank: F1 0.820; tc-F1 0.825; typecheck 0.778.
- Consensus+cycle sweep: best all-F1 0.827 at alpha 0.0 beta 3.0 (tc-F1 0.825).
- Learned reranker (CV): F1 0.823; tc-F1 0.823; typecheck 1.000.
- Oracle headroom: baseline F1 0.788 → oracle 0.924; oracle-tc 0.923.
- Cycle/F1 correlation: pearson avg 0.072, spearman avg -0.042.
- Typecheck signal: tc candidate F1 0.774 vs non-tc 0.740; max tc F1 0.923 vs max non-tc 0.868.
- Consensus k-sweep (all candidates): k=4 F1 0.807, k=8 F1 0.817, k=16 F1 0.819, k=32 F1 0.813, k=64 F1 0.835.
- Consensus all vs tc-only: all-consensus F1 0.820 vs tc-only 0.825; tc-only wins 22.2%.

### High-temp + 64 candidates (temp 0.8, start-index 1080)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1080, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.784
  - best_cycle F1 0.736 (tc-F1 0.786)
  - best_lenpen F1 0.741 (tc-F1 0.784)
- Rerank sweep: best all-F1 0.761 at alpha 0.005 (tc-F1 0.789; tc-rate 1.000).
- Consensus rerank: F1 0.818; tc-F1 0.825; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.815 at alpha 0.001 beta 1.0 (tc-F1 0.827).
- Learned reranker (CV): F1 0.814; tc-F1 0.814; typecheck 1.000.
- Oracle headroom: baseline F1 0.784 → oracle 0.933; oracle-tc 0.926.
- Cycle/F1 correlation: pearson avg 0.197, spearman avg 0.086.
- Typecheck signal: tc candidate F1 0.777 vs non-tc 0.710; max tc F1 0.926 vs max non-tc 0.874.
- Consensus k-sweep (all candidates): k=4 F1 0.808, k=8 F1 0.813, k=16 F1 0.831, k=32 F1 0.814, k=64 F1 0.756.
- Consensus all vs tc-only: all-consensus F1 0.818 vs tc-only 0.825; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1100)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1100, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.766
  - best_cycle F1 0.737 (tc-F1 0.756)
  - best_lenpen F1 0.723 (tc-F1 0.759)
- Rerank sweep: best all-F1 0.737 at alpha 0.0 (tc-F1 0.756; tc-rate 1.000).
- Consensus rerank: F1 0.801; tc-F1 0.802; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.799 at alpha 0.005 beta 3.0 (tc-F1 0.794).
- Learned reranker (CV): F1 0.798; tc-F1 0.798; typecheck 1.000.
- Oracle headroom: baseline F1 0.766 → oracle 0.903; oracle-tc 0.885.
- Cycle/F1 correlation: pearson avg 0.305, spearman avg 0.167.
- Typecheck signal: tc candidate F1 0.765 vs non-tc 0.698; max tc F1 0.885 vs max non-tc 0.793.
- Consensus k-sweep (all candidates): k=4 F1 0.777, k=8 F1 0.792, k=16 F1 0.801, k=32 F1 0.785, k=64 F1 0.798.
- Consensus all vs tc-only: all-consensus F1 0.801 vs tc-only 0.802; tc-only wins 15.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1120)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1120, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.803
  - best_cycle F1 0.740 (tc-F1 0.764)
  - best_lenpen F1 0.745 (tc-F1 0.784)
- Rerank sweep: best all-F1 0.745 at alpha 0.001 (tc-F1 0.784; tc-rate 1.000).
- Consensus rerank: F1 0.862; tc-F1 0.860; typecheck 0.944.
- Consensus+cycle sweep: best all-F1 0.847 at alpha 0.0 beta 3.0 (tc-F1 0.844).
- Learned reranker (CV): F1 0.863; tc-F1 0.863; typecheck 1.000.
- Oracle headroom: baseline F1 0.803 → oracle 0.933; oracle-tc 0.912.
- Cycle/F1 correlation: pearson avg 0.328, spearman avg 0.199.
- Typecheck signal: tc candidate F1 0.807 vs non-tc 0.660; max tc F1 0.912 vs max non-tc 0.849.
- Consensus k-sweep (all candidates): k=4 F1 0.817, k=8 F1 0.842, k=16 F1 0.843, k=32 F1 0.860, k=64 F1 0.875.
- Consensus all vs tc-only: all-consensus F1 0.862 vs tc-only 0.860; tc-only wins 5.6%.

### High-temp + 64 candidates (temp 0.8, start-index 1140)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1140, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.748
  - best_cycle F1 0.726 (tc-F1 0.744)
  - best_lenpen F1 0.729 (tc-F1 0.753)
- Rerank sweep: best all-F1 0.736 at alpha 0.002 (tc-F1 0.754; tc-rate 0.900).
- Consensus rerank: F1 0.767; tc-F1 0.773; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.796 at alpha 0.0 beta 2.0 (tc-F1 0.801).
- Learned reranker (CV): F1 0.792; tc-F1 0.792; typecheck 0.900.
- Oracle headroom: baseline F1 0.748 → oracle 0.906; oracle-tc 0.886.
- Cycle/F1 correlation: pearson avg 0.179, spearman avg 0.154.
- Typecheck signal: tc candidate F1 0.767 vs non-tc 0.654; max tc F1 0.904 vs max non-tc 0.821.
- Consensus k-sweep (all candidates): k=4 F1 0.765, k=8 F1 0.780, k=16 F1 0.766, k=32 F1 0.774, k=64 F1 0.857.
- Consensus all vs tc-only: all-consensus F1 0.767 vs tc-only 0.773; tc-only wins 16.7%.

### High-temp + 64 candidates (temp 0.8, start-index 1160)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1160, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.792
  - best_cycle F1 0.756 (tc-F1 0.748)
  - best_lenpen F1 0.756 (tc-F1 0.748)
- Rerank sweep: best all-F1 0.756 at alpha 0.0 (tc-F1 0.748; tc-rate 1.000).
- Consensus rerank: F1 0.816; tc-F1 0.812; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.816 at alpha 0.001 beta 3.0 (tc-F1 0.818).
- Learned reranker (CV): F1 0.816; tc-F1 0.818; typecheck 0.842.
- Oracle headroom: baseline F1 0.792 → oracle 0.941; oracle-tc 0.935.
- Cycle/F1 correlation: pearson avg 0.337, spearman avg 0.154.
- Typecheck signal: tc candidate F1 0.763 vs non-tc 0.730; max tc F1 0.935 vs max non-tc 0.840.
- Consensus k-sweep (all candidates): k=4 F1 0.799, k=8 F1 0.811, k=16 F1 0.817, k=32 F1 0.823, k=64 F1 0.683.
- Consensus all vs tc-only: all-consensus F1 0.816 vs tc-only 0.812; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1180)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1180, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.737
  - best_cycle F1 0.744 (tc-F1 0.761)
  - best_lenpen F1 0.747 (tc-F1 0.757)
- Rerank sweep: best all-F1 0.747 at alpha 0.001 (tc-F1 0.757; tc-rate 1.000).
- Consensus rerank: F1 0.803; tc-F1 0.795; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.836 at alpha 0.002 beta 2.0 (tc-F1 0.821).
- Learned reranker (CV): F1 0.797; tc-F1 0.797; typecheck 1.000.
- Oracle headroom: baseline F1 0.737 → oracle 0.917; oracle-tc 0.901.
- Cycle/F1 correlation: pearson avg 0.234, spearman avg 0.161.
- Typecheck signal: tc candidate F1 0.774 vs non-tc 0.676; max tc F1 0.901 vs max non-tc 0.839.
- Consensus k-sweep (all candidates): k=4 F1 0.778, k=8 F1 0.791, k=16 F1 0.815, k=32 F1 0.798, k=64 F1 0.820.
- Consensus all vs tc-only: all-consensus F1 0.803 vs tc-only 0.795; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1200)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1200, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.824
  - best_cycle F1 0.772 (tc-F1 0.799)
  - best_lenpen F1 0.762 (tc-F1 0.768)
- Rerank sweep: best all-F1 0.772 at alpha 0.0 (tc-F1 0.799; tc-rate 1.000).
- Consensus rerank: F1 0.855; tc-F1 0.864; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.836 at alpha 0.002 beta 2.0 (tc-F1 0.848).
- Learned reranker (CV): F1 0.851; tc-F1 0.857; typecheck 0.950.
- Oracle headroom: baseline F1 0.824 → oracle 0.938; oracle-tc 0.929.
- Cycle/F1 correlation: pearson avg 0.193, spearman avg 0.061.
- Typecheck signal: tc candidate F1 0.814 vs non-tc 0.713; max tc F1 0.929 vs max non-tc 0.825.
- Consensus k-sweep (all candidates): k=4 F1 0.837, k=8 F1 0.849, k=16 F1 0.850, k=32 F1 0.840, k=64 F1 0.759.
- Consensus all vs tc-only: all-consensus F1 0.855 vs tc-only 0.864; tc-only wins 15.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1220)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1220, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.752
  - best_cycle F1 0.735 (tc-F1 0.769)
  - best_lenpen F1 0.715 (tc-F1 0.730)
- Rerank sweep: best all-F1 0.735 at alpha 0.0 (tc-F1 0.769; tc-rate 0.947).
- Consensus rerank: F1 0.802; tc-F1 0.805; typecheck 0.737.
- Consensus+cycle sweep: best all-F1 0.802 at alpha 0.0 beta 3.0 (tc-F1 0.812).
- Learned reranker (CV): F1 0.808; tc-F1 0.808; typecheck 0.947.
- Oracle headroom: baseline F1 0.752 → oracle 0.900; oracle-tc 0.870.
- Cycle/F1 correlation: pearson avg 0.293, spearman avg 0.192.
- Typecheck signal: tc candidate F1 0.779 vs non-tc 0.649; max tc F1 0.888 vs max non-tc 0.812.
- Consensus k-sweep (all candidates): k=4 F1 0.778, k=8 F1 0.795, k=16 F1 0.795, k=32 F1 0.794, k=64 F1 0.781.
- Consensus all vs tc-only: all-consensus F1 0.802 vs tc-only 0.805; tc-only wins 11.1%.

### High-temp + 64 candidates (temp 0.8, start-index 1240)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1240, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.758
  - best_cycle F1 0.761 (tc-F1 0.766)
  - best_lenpen F1 0.750 (tc-F1 0.766)
- Rerank sweep: best all-F1 0.761 at alpha 0.0 (tc-F1 0.766; tc-rate 1.000).
- Consensus rerank: F1 0.808; tc-F1 0.810; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.798 at alpha 0.0 beta 3.0 (tc-F1 0.807).
- Learned reranker (CV): F1 0.819; tc-F1 0.814; typecheck 0.950.
- Oracle headroom: baseline F1 0.758 → oracle 0.928; oracle-tc 0.901.
- Cycle/F1 correlation: pearson avg 0.250, spearman avg 0.128.
- Typecheck signal: tc candidate F1 0.787 vs non-tc 0.692; max tc F1 0.901 vs max non-tc 0.817.
- Consensus k-sweep (all candidates): k=4 F1 0.788, k=8 F1 0.803, k=16 F1 0.802, k=32 F1 0.812, k=64 F1 0.801.
- Consensus all vs tc-only: all-consensus F1 0.808 vs tc-only 0.810; tc-only wins 15.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1260)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1260, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.728
  - best_cycle F1 0.711 (tc-F1 0.771)
  - best_lenpen F1 0.720 (tc-F1 0.780)
- Rerank sweep: best all-F1 0.721 at alpha 0.002 (tc-F1 0.782; tc-rate 0.947).
- Consensus rerank: F1 0.804; tc-F1 0.816; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.774 at alpha 0.02 beta 2.0 (tc-F1 0.813).
- Learned reranker (CV): F1 0.814; tc-F1 0.823; typecheck 0.895.
- Learned reranker (+baseline sim): F1 0.813; tc-F1 0.822; baseline weight -0.017.
- Oracle headroom: baseline F1 0.728 → oracle 0.915; oracle-tc 0.907.
- Cycle/F1 correlation: pearson avg 0.182, spearman avg 0.136.
- Typecheck signal: tc candidate F1 0.794 vs non-tc 0.673; max tc F1 0.928 vs max non-tc 0.822.
- Consensus k-sweep (all candidates): k=4 F1 0.790, k=8 F1 0.800, k=16 F1 0.808, k=32 F1 0.809, k=64 F1 0.815.
- Consensus all vs tc-only: all-consensus F1 0.804 vs tc-only 0.825; tc-only wins 22.2%.

### High-temp + 64 candidates (temp 0.8, start-index 1280)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1280, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.746
  - best_cycle F1 0.737 (tc-F1 0.711)
  - best_lenpen F1 0.734 (tc-F1 0.717)
- Rerank sweep: best all-F1 0.737 at alpha 0.0 (tc-F1 0.711; tc-rate 0.947).
- Consensus rerank: F1 0.817; tc-F1 0.809; typecheck 0.737.
- Consensus+cycle sweep: best all-F1 0.814 at alpha 0.001 beta 2.0 (tc-F1 0.802).
- Learned reranker (CV): F1 0.788; tc-F1 0.788; typecheck 0.947.
- Oracle headroom: baseline F1 0.746 → oracle 0.925; oracle-tc 0.893.
- Cycle/F1 correlation: pearson avg 0.208, spearman avg 0.164.
- Typecheck signal: tc candidate F1 0.771 vs non-tc 0.682; max tc F1 0.896 vs max non-tc 0.861.
- Consensus k-sweep (all candidates): k=4 F1 0.792, k=8 F1 0.804, k=16 F1 0.803, k=32 F1 0.795, k=64 F1 0.793.
- Consensus all vs tc-only: all-consensus F1 0.817 vs tc-only 0.809; tc-only wins 11.1%.

### High-temp + 64 candidates (temp 0.8, start-index 1300)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1300, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.792
  - best_cycle F1 0.749 (tc-F1 0.722)
  - best_lenpen F1 0.719 (tc-F1 0.733)
- Rerank sweep: best all-F1 0.749 at alpha 0.0 (tc-F1 0.722; tc-rate 1.000).
- Consensus rerank: F1 0.817; tc-F1 0.820; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.816 at alpha 0.002 beta 3.0 (tc-F1 0.821).
- Learned reranker (CV): F1 0.816; tc-F1 0.823; typecheck 0.684.
- Oracle headroom: baseline F1 0.792 → oracle 0.917; oracle-tc 0.913.
- Cycle/F1 correlation: pearson avg 0.331, spearman avg 0.124.
- Typecheck signal: tc candidate F1 0.770 vs non-tc 0.748; max tc F1 0.913 vs max non-tc 0.829.
- Consensus k-sweep (all candidates): k=4 F1 0.803, k=8 F1 0.810, k=16 F1 0.815, k=32 F1 0.805, k=64 F1 0.874.
- Consensus all vs tc-only: all-consensus F1 0.817 vs tc-only 0.820; tc-only wins 21.1%.

### High-temp + 64 candidates (temp 0.8, start-index 1320)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1320, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.773
  - best_cycle F1 0.783 (tc-F1 0.790)
  - best_lenpen F1 0.797 (tc-F1 0.793)
- Rerank sweep: best all-F1 0.798 at alpha 0.002 (tc-F1 0.793; tc-rate 1.000).
- Consensus rerank: F1 0.836; tc-F1 0.846; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.826 at alpha 0.005 beta 2.0 (tc-F1 0.837).
- Learned reranker (CV): F1 0.835; tc-F1 0.839; typecheck 0.800.
- Oracle headroom: baseline F1 0.773 → oracle 0.931; oracle-tc 0.915.
- Cycle/F1 correlation: pearson avg 0.266, spearman avg 0.128.
- Typecheck signal: tc candidate F1 0.788 vs non-tc 0.747; max tc F1 0.915 vs max non-tc 0.840.
- Consensus k-sweep (all candidates): k=4 F1 0.826, k=8 F1 0.831, k=16 F1 0.830, k=32 F1 0.833, k=64 F1 0.842.
- Consensus all vs tc-only: all-consensus F1 0.836 vs tc-only 0.846; tc-only wins 25.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1340)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1340, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.835
  - best_cycle F1 0.703 (tc-F1 0.735)
  - best_lenpen F1 0.703 (tc-F1 0.743)
- Rerank sweep: best all-F1 0.706 at alpha 0.002 (tc-F1 0.743; tc-rate 1.000).
- Consensus rerank: F1 0.856; tc-F1 0.855; typecheck 0.889.
- Consensus+cycle sweep: best all-F1 0.827 at alpha 0.0 beta 3.0 (tc-F1 0.831).
- Learned reranker (CV): F1 0.851; tc-F1 0.849; typecheck 0.889.
- Oracle headroom: baseline F1 0.835 → oracle 0.936; oracle-tc 0.936.
- Cycle/F1 correlation: pearson avg 0.048, spearman avg -0.015.
- Typecheck signal: tc candidate F1 0.794 vs non-tc 0.764; max tc F1 0.936 vs max non-tc 0.874.
- Consensus k-sweep (all candidates): k=4 F1 0.827, k=8 F1 0.843, k=16 F1 0.853, k=32 F1 0.856, k=64 F1 0.803.
- Consensus all vs tc-only: all-consensus F1 0.856 vs tc-only 0.855; tc-only wins 16.7%.

### High-temp + 64 candidates (temp 0.8, start-index 1360)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1360, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.731
  - best_cycle F1 0.716 (tc-F1 0.743)
  - best_lenpen F1 0.719 (tc-F1 0.740)
- Rerank sweep: best all-F1 0.719 at alpha 0.001 (tc-F1 0.740; tc-rate 1.000).
- Consensus rerank: F1 0.784; tc-F1 0.787; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.769 at alpha 0.0 beta 3.0 (tc-F1 0.774).
- Learned reranker (CV): F1 0.787; tc-F1 0.786; typecheck 0.900.
- Oracle headroom: baseline F1 0.731 → oracle 0.912; oracle-tc 0.860.
- Cycle/F1 correlation: pearson avg 0.186, spearman avg 0.109.
- Typecheck signal: tc candidate F1 0.733 vs non-tc 0.690; max tc F1 0.860 vs max non-tc 0.803.
- Consensus k-sweep (all candidates): k=4 F1 0.758, k=8 F1 0.767, k=16 F1 0.774, k=32 F1 0.776, k=64 F1 0.812.
- Consensus all vs tc-only: all-consensus F1 0.784 vs tc-only 0.787; tc-only wins 30.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1380)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1380, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.768
  - best_cycle F1 0.761 (tc-F1 0.735)
  - best_lenpen F1 0.780 (tc-F1 0.766)
- Rerank sweep: best all-F1 0.780 at alpha 0.001 (tc-F1 0.766; tc-rate 1.000).
- Consensus rerank: F1 0.832; tc-F1 0.820; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.826 at alpha 0.0 beta 3.0 (tc-F1 0.817).
- Learned reranker (CV): F1 0.836; tc-F1 0.836; typecheck 1.000.
- Oracle headroom: baseline F1 0.768 → oracle 0.927; oracle-tc 0.917.
- Cycle/F1 correlation: pearson avg 0.294, spearman avg 0.107.
- Typecheck signal: tc candidate F1 0.792 vs non-tc 0.670; max tc F1 0.917 vs max non-tc 0.809.
- Consensus k-sweep (all candidates): k=4 F1 0.811, k=8 F1 0.821, k=16 F1 0.827, k=32 F1 0.815, k=64 F1 0.768.
- Consensus all vs tc-only: all-consensus F1 0.832 vs tc-only 0.820; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1400)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1400, 20 requested (17 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.832
  - best_cycle F1 0.768 (tc-F1 0.773)
  - best_lenpen F1 0.812 (tc-F1 0.818)
- Rerank sweep: best all-F1 0.812 at alpha 0.001 (tc-F1 0.818; tc-rate 1.000).
- Consensus rerank: F1 0.846; tc-F1 0.850; typecheck 0.706.
- Consensus+cycle sweep: best all-F1 0.850 at alpha 0.001 beta 3.0 (tc-F1 0.847).
- Learned reranker (CV): F1 0.849; tc-F1 0.851; typecheck 0.941.
- Oracle headroom: baseline F1 0.832 → oracle 0.938; oracle-tc 0.932.
- Cycle/F1 correlation: pearson avg 0.509, spearman avg 0.313.
- Typecheck signal: tc candidate F1 0.787 vs non-tc 0.739; max tc F1 0.932 vs max non-tc 0.822.
- Consensus k-sweep (all candidates): k=4 F1 0.830, k=8 F1 0.838, k=16 F1 0.841, k=32 F1 0.838, k=64 F1 0.935 (k=64 only 30 samples).
- Consensus all vs tc-only: all-consensus F1 0.846 vs tc-only 0.850; tc-only wins 11.8%.

### High-temp + 64 candidates (temp 0.8, start-index 1420)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1420, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.816
  - best_cycle F1 0.763 (tc-F1 0.801)
  - best_lenpen F1 0.752 (tc-F1 0.793)
- Rerank sweep: best all-F1 0.763 at alpha 0.0 (tc-F1 0.801; tc-rate 1.000).
- Consensus rerank: F1 0.875; tc-F1 0.884; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.867 at alpha 0.01 beta 3.0 (tc-F1 0.874).
- Learned reranker (CV): F1 0.879; tc-F1 0.879; typecheck 1.000.
- Oracle headroom: baseline F1 0.816 → oracle 0.942; oracle-tc 0.934.
- Cycle/F1 correlation: pearson avg 0.177, spearman avg 0.071.
- Typecheck signal: tc candidate F1 0.844 vs non-tc 0.760; max tc F1 0.934 vs max non-tc 0.853.
- Consensus k-sweep (all candidates): k=4 F1 0.857, k=8 F1 0.866, k=16 F1 0.872, k=32 F1 0.883, k=64 F1 0.888.
- Consensus all vs tc-only: all-consensus F1 0.875 vs tc-only 0.884; tc-only wins 15.8%.

### High-temp + 64 candidates (temp 0.8, start-index 1440)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1440, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.788
  - best_cycle F1 0.728 (tc-F1 0.735)
  - best_lenpen F1 0.750 (tc-F1 0.748)
- Rerank sweep: best all-F1 0.750 at alpha 0.001 (tc-F1 0.748; tc-rate 1.000).
- Consensus rerank: F1 0.828; tc-F1 0.771; typecheck 0.722.
- Consensus+cycle sweep: best all-F1 0.814 at alpha 0.005 beta 3.0 (tc-F1 0.783).
- Learned reranker (CV): F1 0.831; tc-F1 0.792; typecheck 0.889.
- Oracle headroom: baseline F1 0.788 → oracle 0.917; oracle-tc 0.858.
- Cycle/F1 correlation: pearson avg 0.328, spearman avg 0.206.
- Typecheck signal: tc candidate F1 0.784 vs non-tc 0.683; max tc F1 0.858 vs max non-tc 0.853.
- Consensus k-sweep (all candidates): k=4 F1 0.793, k=8 F1 0.808, k=16 F1 0.818, k=32 F1 0.826, k=64 F1 0.820.
- Consensus all vs tc-only: all-consensus F1 0.828 vs tc-only 0.755; tc-only wins 5.6%.

### High-temp + 64 candidates (temp 0.8, start-index 1460)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1460, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.755
  - best_cycle F1 0.756 (tc-F1 0.765)
  - best_lenpen F1 0.755 (tc-F1 0.758)
- Rerank sweep: best all-F1 0.756 at alpha 0.0 (tc-F1 0.765; tc-rate 1.000).
- Consensus rerank: F1 0.820; tc-F1 0.820; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.818 at alpha 0.002 beta 3.0 (tc-F1 0.813).
- Learned reranker (CV): F1 0.801; tc-F1 0.791; typecheck 0.900.
- Oracle headroom: baseline F1 0.755 → oracle 0.909; oracle-tc 0.891.
- Cycle/F1 correlation: pearson avg 0.264, spearman avg 0.164.
- Typecheck signal: tc candidate F1 0.767 vs non-tc 0.706; max tc F1 0.891 vs max non-tc 0.808.
- Consensus k-sweep (all candidates): k=4 F1 0.785, k=8 F1 0.793, k=16 F1 0.799, k=32 F1 0.785, k=64 F1 0.848.
- Consensus all vs tc-only: all-consensus F1 0.820 vs tc-only 0.820; tc-only wins 10.0%.

### High-temp + 64 candidates (temp 0.8, start-index 1480)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1480, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.775
  - best_cycle F1 0.728 (tc-F1 0.734)
  - best_lenpen F1 0.728 (tc-F1 0.702)
- Rerank sweep: best all-F1 0.728 at alpha 0.0 (tc-F1 0.734; tc-rate 1.000).
- Consensus rerank: F1 0.808; tc-F1 0.799; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.798 at alpha 0.001 beta 3.0 (tc-F1 0.797).
- Learned reranker (CV, no-cycle): F1 0.807; tc-F1 0.807; typecheck 1.000.
- Oracle headroom: baseline F1 0.775 → oracle 0.920; oracle-tc 0.898.
- Cycle/F1 correlation: pearson avg 0.221, spearman avg 0.062.
- Typecheck signal: tc candidate F1 0.760 vs non-tc 0.671; max tc F1 0.898 vs max non-tc 0.854.
- Consensus k-sweep (all candidates): k=4 F1 0.788, k=8 F1 0.800, k=16 F1 0.798, k=32 F1 0.785, k=64 F1 0.822.
- Consensus all vs tc-only: all-consensus F1 0.808 vs tc-only 0.799; tc-only wins 10.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1500)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1500, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.788
  - best_cycle F1 0.767 (tc-F1 0.768)
  - best_lenpen F1 0.769 (tc-F1 0.781)
- Rerank sweep: best all-F1 0.769 at alpha 0.001 (tc-F1 0.781; tc-rate 0.950).
- Consensus rerank: F1 0.825; tc-F1 0.816; typecheck 0.700.
- Consensus+cycle sweep: best all-F1 0.823 at alpha 0.005 beta 2.0 (tc-F1 0.835).
- Learned reranker (CV, no-cycle): F1 0.818; tc-F1 0.818; typecheck 0.950.
- Oracle headroom: baseline F1 0.788 → oracle 0.931; oracle-tc 0.922.
- Cycle/F1 correlation: pearson avg 0.381, spearman avg 0.279.
- Typecheck signal: tc candidate F1 0.780 vs non-tc 0.676; max tc F1 0.935 vs max non-tc 0.835.
- Consensus k-sweep (all candidates): k=4 F1 0.792, k=8 F1 0.801, k=16 F1 0.815, k=32 F1 0.814, k=64 F1 0.822.
- Consensus all vs tc-only: all-consensus F1 0.825 vs tc-only 0.816; tc-only wins 15.8%.

### High-temp + 64 candidates (temp 0.8, start-index 1520)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1520, 20 requested (17 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.779
  - best_cycle F1 0.763 (tc-F1 0.775)
  - best_lenpen F1 0.778 (tc-F1 0.736)
- Rerank sweep: best all-F1 0.782 at alpha 0.002 (tc-F1 0.744; tc-rate 1.000).
- Consensus rerank: F1 0.863; tc-F1 0.867; typecheck 0.706.
- Consensus+cycle sweep: best all-F1 0.872 at alpha 0.0 beta 3.0 (tc-F1 0.874).
- Learned reranker (CV, no-cycle): F1 0.862; tc-F1 0.871; typecheck 0.588.
- Oracle headroom: baseline F1 0.779 → oracle 0.939; oracle-tc 0.929.
- Cycle/F1 correlation: pearson avg 0.176, spearman avg 0.096.
- Typecheck signal: tc candidate F1 0.804 vs non-tc 0.784; max tc F1 0.929 vs max non-tc 0.880.
- Consensus k-sweep (all candidates): k=4 F1 0.835, k=8 F1 0.853, k=16 F1 0.864, k=32 F1 0.866, k=64 F1 0.912.
- Consensus all vs tc-only: all-consensus F1 0.863 vs tc-only 0.867; tc-only wins 23.5%.

### High-temp + 64 candidates (temp 0.8, start-index 1540)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1540, 20 requested (18 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.789
  - best_cycle F1 0.775 (tc-F1 0.779)
  - best_lenpen F1 0.767 (tc-F1 0.779)
- Rerank sweep: best all-F1 0.775 at alpha 0.0 (tc-F1 0.779; tc-rate 1.000).
- Consensus rerank: F1 0.833; tc-F1 0.827; typecheck 0.889.
- Consensus+cycle sweep: best all-F1 0.829 at alpha 0.0 beta 3.0 (tc-F1 0.838).
- Learned reranker (CV, no-cycle): F1 0.849; tc-F1 0.845; typecheck 0.833.
- Oracle headroom: baseline F1 0.789 → oracle 0.946; oracle-tc 0.926.
- Cycle/F1 correlation: pearson avg 0.265, spearman avg 0.110.
- Typecheck signal: tc candidate F1 0.787 vs non-tc 0.729; max tc F1 0.926 vs max non-tc 0.853.
- Consensus k-sweep (all candidates): k=4 F1 0.812, k=8 F1 0.824, k=16 F1 0.832, k=32 F1 0.832, k=64 F1 0.890.
- Consensus all vs tc-only: all-consensus F1 0.833 vs tc-only 0.827; tc-only wins 5.6%.

### High-temp + 64 candidates (temp 0.8, start-index 1560)

- Run: num_candidates 64, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1560, 20 requested (17 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.796
  - best_cycle F1 0.806 (tc-F1 0.802)
  - best_lenpen F1 0.802 (tc-F1 0.818)
- Rerank sweep: best all-F1 0.812 at alpha 0.005 (tc-F1 0.827; tc-rate 0.824).
- Consensus rerank: F1 0.832; tc-F1 0.826; typecheck 0.882.
- Consensus+cycle sweep: best all-F1 0.859 at alpha 0.0 beta 1.0 (tc-F1 0.851).
- Learned reranker (CV, no-cycle): F1 0.823; tc-F1 0.823; typecheck 1.000.
- Oracle headroom: baseline F1 0.796 → oracle 0.939; oracle-tc 0.936.
- Cycle/F1 correlation: pearson avg 0.259, spearman avg 0.207.
- Typecheck signal: tc candidate F1 0.806 vs non-tc 0.707; max tc F1 0.936 vs max non-tc 0.840.
- Consensus k-sweep (all candidates): k=4 F1 0.819, k=8 F1 0.825, k=16 F1 0.831, k=32 F1 0.817, k=64 F1 0.787.
- Consensus all vs tc-only: all-consensus F1 0.832 vs tc-only 0.826; tc-only wins 5.9%.

### High-temp + 128 candidates (temp 0.8, start-index 1580)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1580, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.808
  - best_cycle F1 0.752 (tc-F1 0.818)
  - best_lenpen F1 0.748 (tc-F1 0.777)
- Rerank sweep: best all-F1 0.752 at alpha 0.0 (tc-F1 0.818; tc-rate 0.400).
- Consensus rerank: F1 0.875; tc-F1 0.864; typecheck 0.900.
- Consensus+cycle sweep: best all-F1 0.858 at alpha 0.001 beta 3.0 (tc-F1 0.861).
- Learned reranker (CV, no-cycle): F1 0.874; tc-F1 0.865; typecheck 0.950.
- Oracle headroom: baseline F1 0.808 → oracle 0.947; oracle-tc 0.937.
- Cycle/F1 correlation: pearson avg 0.277, spearman avg 0.195.
- Typecheck signal: tc candidate F1 0.813 vs non-tc 0.741; max tc F1 0.937 vs max non-tc 0.885.
- Consensus k-sweep (all candidates): k=4 F1 0.834, k=8 F1 0.848, k=16 F1 0.860, k=32 F1 0.874, k=64 F1 0.863, k=128 F1 0.810.
- Consensus all vs tc-only: all-consensus F1 0.875 vs tc-only 0.864; tc-only wins 20.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1600)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1600, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.788
  - best_cycle F1 0.751 (tc-F1 0.795)
  - best_lenpen F1 0.779 (tc-F1 0.796)
- Rerank sweep: best all-F1 0.779 at alpha 0.001 (tc-F1 0.796; tc-rate 0.632).
- Consensus rerank: F1 0.872; tc-F1 0.863; typecheck 0.737.
- Consensus+cycle sweep: best all-F1 0.856 at alpha 0.005 beta 2.0 (tc-F1 0.851).
- Learned reranker (CV, no-cycle): F1 0.874; tc-F1 0.859; typecheck 0.842.
- Oracle headroom: baseline F1 0.788 → oracle 0.957; oracle-tc 0.935.
- Cycle/F1 correlation: pearson avg 0.320, spearman avg 0.208.
- Typecheck signal: tc candidate F1 0.843 vs non-tc 0.760; max tc F1 0.935 vs max non-tc 0.873.
- Consensus k-sweep (all candidates): k=4 F1 0.853, k=8 F1 0.864, k=16 F1 0.873, k=32 F1 0.875, k=64 F1 0.872, k=128 F1 0.000 (no records).
- Consensus all vs tc-only: all-consensus F1 0.872 vs tc-only 0.863; tc-only wins 36.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1620)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1620, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.677
  - best_cycle F1 0.733 (tc-F1 0.740)
  - best_lenpen F1 0.721 (tc-F1 0.740)
- Rerank sweep: best all-F1 0.733 at alpha 0.0 (tc-F1 0.740; tc-rate 0.350).
- Consensus rerank: F1 0.774; tc-F1 0.791; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.773 at alpha 0.0 beta 2.0 (tc-F1 0.769).
- Learned reranker (CV, no-cycle): F1 0.789; tc-F1 0.789; typecheck 0.950.
- Oracle headroom: baseline F1 0.677 → oracle 0.912; oracle-tc 0.897.
- Cycle/F1 correlation: pearson avg 0.423, spearman avg 0.221.
- Typecheck signal: tc candidate F1 0.731 vs non-tc 0.656; max tc F1 0.908 vs max non-tc 0.846.
- Consensus k-sweep (all candidates): k=4 F1 0.759, k=8 F1 0.769, k=16 F1 0.772, k=32 F1 0.787, k=64 F1 0.764, k=128 F1 0.777.
- Consensus all vs tc-only: all-consensus F1 0.774 vs tc-only 0.791; tc-only wins 26.3%.

### High-temp + 128 candidates (temp 0.8, start-index 1640)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1640, 20 requested (20 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.796
  - best_cycle F1 0.747 (tc-F1 0.799)
  - best_lenpen F1 0.738 (tc-F1 0.808)
- Rerank sweep: best all-F1 0.747 at alpha 0.0 (tc-F1 0.799; tc-rate 0.500).
- Consensus rerank: F1 0.861; tc-F1 0.868; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.846 at alpha 0.01 beta 3.0 (tc-F1 0.863).
- Learned reranker (CV, no-cycle): F1 0.862; tc-F1 0.869; typecheck 0.900.
- Oracle headroom: baseline F1 0.796 → oracle 0.929; oracle-tc 0.925.
- Cycle/F1 correlation: pearson avg 0.337, spearman avg 0.170.
- Typecheck signal: tc candidate F1 0.799 vs non-tc 0.746; max tc F1 0.925 vs max non-tc 0.849.
- Consensus k-sweep (all candidates): k=4 F1 0.832, k=8 F1 0.847, k=16 F1 0.853, k=32 F1 0.863, k=64 F1 0.860, k=128 F1 0.815.
- Consensus all vs tc-only: all-consensus F1 0.861 vs tc-only 0.868; tc-only wins 20.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1660)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1660, 20 requested (19 unique), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.730
  - best_cycle F1 0.737 (tc-F1 0.781)
  - best_lenpen F1 0.730 (tc-F1 0.775)
- Rerank sweep: best all-F1 0.737 at alpha 0.0 (tc-F1 0.781; tc-rate 1.000).
- Consensus rerank: F1 0.839; tc-F1 0.865; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.836 at alpha 0.0 beta 3.0 (tc-F1 0.843).
- Learned reranker (CV, no-cycle): F1 0.839; tc-F1 0.850; typecheck 0.947.
- Oracle headroom: baseline F1 0.730 → oracle 0.935; oracle-tc 0.924.
- Cycle/F1 correlation: pearson avg 0.304, spearman avg 0.140.
- Typecheck signal: tc candidate F1 0.780 vs non-tc 0.690; max tc F1 0.924 vs max non-tc 0.856.
- Consensus k-sweep (all candidates): k=4 F1 0.794, k=8 F1 0.817, k=16 F1 0.832, k=32 F1 0.839, k=64 F1 0.841.
- Consensus all vs tc-only: all-consensus F1 0.839 vs tc-only 0.865; tc-only wins 36.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1680)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1680, 20 requested (17 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.802
  - best_cycle F1 0.783 (tc-F1 0.802)
  - best_lenpen F1 0.782 (tc-F1 0.801)
- Rerank sweep: best all-F1 0.783 at alpha 0.0 (tc-F1 0.802; tc-rate 1.000).
- Consensus rerank: F1 0.836; tc-F1 0.838; typecheck 0.824.
- Consensus+cycle sweep: best all-F1 0.837 at alpha 0.0 beta 2.0 (tc-F1 0.840).
- Learned reranker (CV, no-cycle): F1 0.868; tc-F1 0.868; typecheck 1.000.
- Oracle headroom: baseline F1 0.802 → oracle 0.944; oracle-tc 0.943.
- Cycle/F1 correlation: pearson avg 0.224, spearman avg 0.129.
- Typecheck signal: tc candidate F1 0.806 vs non-tc 0.725; max tc F1 0.943 vs max non-tc 0.851.
- Consensus k-sweep (all candidates): k=4 F1 0.818, k=8 F1 0.835, k=16 F1 0.835, k=32 F1 0.836, k=64 F1 0.834.
- Consensus all vs tc-only: all-consensus F1 0.836 vs tc-only 0.838; tc-only wins 11.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1700)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1700, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.767
  - best_cycle F1 0.702 (tc-F1 0.752)
  - best_lenpen F1 0.715 (tc-F1 0.752)
- Rerank sweep: best all-F1 0.724 at alpha 0.005 (tc-F1 0.687; tc-rate 0.526).
- Consensus rerank: F1 0.799; tc-F1 0.804; typecheck 0.737.
- Consensus+cycle sweep: best all-F1 0.809 at alpha 0.0 beta 3.0 (tc-F1 0.808).
- Learned reranker (CV, no-cycle): F1 0.821; tc-F1 0.821; typecheck 1.000.
- Oracle headroom: baseline F1 0.767 → oracle 0.944; oracle-tc 0.928.
- Cycle/F1 correlation: pearson avg 0.194, spearman avg 0.048.
- Typecheck signal: tc candidate F1 0.785 vs non-tc 0.678; max tc F1 0.928 vs max non-tc 0.868.
- Consensus k-sweep (all candidates): k=4 F1 0.786, k=8 F1 0.793, k=16 F1 0.794, k=32 F1 0.802, k=64 F1 0.797.
- Consensus all vs tc-only: all-consensus F1 0.799 vs tc-only 0.804; tc-only wins 15.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1720)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1720, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.770
  - best_cycle F1 0.770 (tc-F1 0.783)
  - best_lenpen F1 0.786 (tc-F1 0.792)
- Rerank sweep: best all-F1 0.786 at alpha 0.001 (tc-F1 0.792; tc-rate 0.556).
- Consensus rerank: F1 0.852; tc-F1 0.858; typecheck 0.778.
- Consensus+cycle sweep: best all-F1 0.845 at alpha 0.001 beta 3.0 (tc-F1 0.844).
- Learned reranker (CV, no-cycle): F1 0.849; tc-F1 0.852; typecheck 0.944.
- Oracle headroom: baseline F1 0.770 → oracle 0.925; oracle-tc 0.920.
- Cycle/F1 correlation: pearson avg 0.456, spearman avg 0.251.
- Typecheck signal: tc candidate F1 0.779 vs non-tc 0.733; max tc F1 0.920 vs max non-tc 0.857.
- Consensus k-sweep (all candidates): k=4 F1 0.815, k=8 F1 0.831, k=16 F1 0.837, k=32 F1 0.843, k=64 F1 0.821.
- Consensus all vs tc-only: all-consensus F1 0.852 vs tc-only 0.858; tc-only wins 22.2%.

### High-temp + 128 candidates (temp 0.8, start-index 1740)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1740, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.810
  - best_cycle F1 0.748 (tc-F1 0.771)
  - best_lenpen F1 0.734 (tc-F1 0.764)
- Rerank sweep: best all-F1 0.748 at alpha 0.0 (tc-F1 0.771; tc-rate 1.000).
- Consensus rerank: F1 0.825; tc-F1 0.826; typecheck 0.833.
- Consensus+cycle sweep: best all-F1 0.833 at alpha 0.0 beta 3.0 (tc-F1 0.818).
- Learned reranker (CV, no-cycle): F1 0.824; tc-F1 0.813; typecheck 0.833.
- Oracle headroom: baseline F1 0.810 → oracle 0.939; oracle-tc 0.917.
- Cycle/F1 correlation: pearson avg 0.314, spearman avg 0.196.
- Typecheck signal: tc candidate F1 0.789 vs non-tc 0.734; max tc F1 0.917 vs max non-tc 0.861.
- Consensus k-sweep (all candidates): k=4 F1 0.815, k=8 F1 0.825, k=16 F1 0.832, k=32 F1 0.825, k=64 F1 0.825.
- Consensus all vs tc-only: all-consensus F1 0.825 vs tc-only 0.793; tc-only wins 27.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1760)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1760, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.781
  - best_cycle F1 0.665 (tc-F1 0.748)
  - best_lenpen F1 0.667 (tc-F1 0.742)
- Rerank sweep: best all-F1 0.675 at alpha 0.002 (tc-F1 0.747; tc-rate 0.389).
- Consensus rerank: F1 0.843; tc-F1 0.824; typecheck 0.722.
- Consensus+cycle sweep: best all-F1 0.804 at alpha 0.01 beta 3.0 (tc-F1 0.820).
- Learned reranker (CV, no-cycle): F1 0.841; tc-F1 0.836; typecheck 0.944.
- Oracle headroom: baseline F1 0.781 → oracle 0.937; oracle-tc 0.929.
- Cycle/F1 correlation: pearson avg 0.300, spearman avg 0.192.
- Typecheck signal: tc candidate F1 0.756 vs non-tc 0.731; max tc F1 0.929 vs max non-tc 0.873.
- Consensus k-sweep (all candidates): k=4 F1 0.801, k=8 F1 0.813, k=16 F1 0.824, k=32 F1 0.828, k=64 F1 0.817.
- Consensus all vs tc-only: all-consensus F1 0.843 vs tc-only 0.824; tc-only wins 22.2%.

### High-temp + 128 candidates (temp 0.8, start-index 1780)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1780, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.823
  - best_cycle F1 0.771 (tc-F1 0.811)
  - best_lenpen F1 0.782 (tc-F1 0.822)
- Rerank sweep: best all-F1 0.782 at alpha 0.001 (tc-F1 0.822; tc-rate 0.750).
- Consensus rerank: F1 0.872; tc-F1 0.876; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.881 at alpha 0.0 beta 3.0 (tc-F1 0.884).
- Learned reranker (CV, no-cycle): F1 0.878; tc-F1 0.878; typecheck 0.950.
- Oracle headroom: baseline F1 0.823 → oracle 0.953; oracle-tc 0.946.
- Cycle/F1 correlation: pearson avg 0.307, spearman avg 0.196.
- Typecheck signal: tc candidate F1 0.825 vs non-tc 0.768; max tc F1 0.946 vs max non-tc 0.892.
- Consensus k-sweep (all candidates): k=4 F1 0.851, k=8 F1 0.855, k=16 F1 0.863, k=32 F1 0.857, k=64 F1 0.864.
- Consensus all vs tc-only: all-consensus F1 0.872 vs tc-only 0.876; tc-only wins 15.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1800)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1800, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.822
  - best_cycle F1 0.772 (tc-F1 0.804)
  - best_lenpen F1 0.776 (tc-F1 0.812)
- Rerank sweep: best all-F1 0.776 at alpha 0.001 (tc-F1 0.812; tc-rate 0.632).
- Consensus rerank: F1 0.860; tc-F1 0.872; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.859 at alpha 0.001 beta 3.0 (tc-F1 0.868).
- Learned reranker (CV, no-cycle): F1 0.865; tc-F1 0.874; typecheck 0.947.
- Oracle headroom: baseline F1 0.822 → oracle 0.950; oracle-tc 0.950.
- Cycle/F1 correlation: pearson avg 0.445, spearman avg 0.313.
- Typecheck signal: tc candidate F1 0.810 vs non-tc 0.726; max tc F1 0.950 vs max non-tc 0.871.
- Consensus k-sweep (all candidates): k=4 F1 0.834, k=8 F1 0.846, k=16 F1 0.855, k=32 F1 0.863, k=64 F1 0.860.
- Consensus all vs tc-only: all-consensus F1 0.860 vs tc-only 0.872; tc-only wins 10.5%.

### High-temp + 128 candidates (temp 0.8, start-index 1820)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1820, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.777
  - best_cycle F1 0.762 (tc-F1 0.741)
  - best_lenpen F1 0.760 (tc-F1 0.769)
- Rerank sweep: best all-F1 0.768 at alpha 0.002 (tc-F1 0.768; tc-rate 0.500).
- Consensus rerank: F1 0.843; tc-F1 0.793; typecheck 0.944.
- Consensus+cycle sweep: best all-F1 0.827 at alpha 0.0 beta 3.0 (tc-F1 0.800).
- Learned reranker (CV, no-cycle): F1 0.839; tc-F1 0.793; typecheck 0.556.
- Oracle headroom: baseline F1 0.777 → oracle 0.940; oracle-tc 0.888.
- Cycle/F1 correlation: pearson avg 0.148, spearman avg 0.035.
- Typecheck signal: tc candidate F1 0.798 vs non-tc 0.779; max tc F1 0.888 vs max non-tc 0.873.
- Consensus k-sweep (all candidates): k=4 F1 0.828, k=8 F1 0.839, k=16 F1 0.847, k=32 F1 0.847, k=64 F1 0.844.
- Consensus all vs tc-only: all-consensus F1 0.843 vs tc-only 0.793; tc-only wins 11.1%.

### High-temp + 128 candidates (temp 0.8, start-index 1840)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1840, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.759
  - best_cycle F1 0.736 (tc-F1 0.801)
  - best_lenpen F1 0.748 (tc-F1 0.758)
- Rerank sweep: best all-F1 0.748 at alpha 0.001 (tc-F1 0.758; tc-rate 0.444).
- Consensus rerank: F1 0.853; tc-F1 0.846; typecheck 0.889.
- Consensus+cycle sweep: best all-F1 0.844 at alpha 0.001 beta 1.0 (tc-F1 0.829).
- Learned reranker (CV, no-cycle): F1 0.847; tc-F1 0.847; typecheck 1.000.
- Oracle headroom: baseline F1 0.759 → oracle 0.952; oracle-tc 0.941.
- Cycle/F1 correlation: pearson avg 0.374, spearman avg 0.206.
- Typecheck signal: tc candidate F1 0.802 vs non-tc 0.703; max tc F1 0.941 vs max non-tc 0.851.
- Consensus k-sweep (all candidates): k=4 F1 0.819, k=8 F1 0.829, k=16 F1 0.839, k=32 F1 0.862, k=64 F1 0.859.
- Consensus all vs tc-only: all-consensus F1 0.853 vs tc-only 0.846; tc-only wins 5.6%.

### High-temp + 128 candidates (temp 0.8, start-index 1860)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1860, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.786
  - best_cycle F1 0.718 (tc-F1 0.744)
  - best_lenpen F1 0.707 (tc-F1 0.738)
- Rerank sweep: best all-F1 0.735 at alpha 0.005 (tc-F1 0.763; tc-rate 0.667).
- Consensus rerank: F1 0.823; tc-F1 0.821; typecheck 1.000.
- Consensus+cycle sweep: best all-F1 0.795 at alpha 0.005 beta 3.0 (tc-F1 0.798).
- Learned reranker (CV, no-cycle): F1 0.820; tc-F1 0.820; typecheck 1.000.
- Oracle headroom: baseline F1 0.786 → oracle 0.933; oracle-tc 0.927.
- Cycle/F1 correlation: pearson avg 0.358, spearman avg 0.251.
- Typecheck signal: tc candidate F1 0.759 vs non-tc 0.702; max tc F1 0.927 vs max non-tc 0.860.
- Consensus k-sweep (all candidates): k=4 F1 0.790, k=8 F1 0.805, k=16 F1 0.807, k=32 F1 0.825, k=64 F1 0.828.
- Consensus all vs tc-only: all-consensus F1 0.823 vs tc-only 0.821; tc-only wins 5.6%.

### High-temp + 128 candidates (temp 0.8, start-index 1880)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1880, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.818
  - best_cycle F1 0.791 (tc-F1 0.812)
  - best_lenpen F1 0.792 (tc-F1 0.825)
- Rerank sweep: best all-F1 0.796 at alpha 0.002 (tc-F1 0.829; tc-rate 1.000).
- Consensus rerank: F1 0.865; tc-F1 0.878; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.865 at alpha 0.0 beta 3.0 (tc-F1 0.862).
- Learned reranker (CV, no-cycle): F1 0.873; tc-F1 0.885; typecheck 0.950.
- Oracle headroom: baseline F1 0.818 → oracle 0.951; oracle-tc 0.938.
- Cycle/F1 correlation: pearson avg 0.431, spearman avg 0.250.
- Typecheck signal: tc candidate F1 0.806 vs non-tc 0.721; max tc F1 0.938 vs max non-tc 0.849.
- Consensus k-sweep (all candidates): k=4 F1 0.818, k=8 F1 0.826, k=16 F1 0.847, k=32 F1 0.853, k=64 F1 0.845.
- Consensus all vs tc-only: all-consensus F1 0.865 vs tc-only 0.878; tc-only wins 20.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1900)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1900, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.785
  - best_cycle F1 0.763 (tc-F1 0.814)
  - best_lenpen F1 0.762 (tc-F1 0.774)
- Rerank sweep: best all-F1 0.763 at alpha 0.002 (tc-F1 0.772; tc-rate 1.000).
- Consensus rerank: F1 0.814; tc-F1 0.847; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.817 at alpha 0.0 beta 3.0 (tc-F1 0.855).
- Learned reranker (CV, no-cycle): F1 0.836; tc-F1 0.856; typecheck 0.950.
- Oracle headroom: baseline F1 0.785 → oracle 0.942; oracle-tc 0.940.
- Cycle/F1 correlation: pearson avg 0.308, spearman avg 0.181.
- Typecheck signal: tc candidate F1 0.790 vs non-tc 0.707; max tc F1 0.940 vs max non-tc 0.861.
- Consensus k-sweep (all candidates): k=4 F1 0.798, k=8 F1 0.800, k=16 F1 0.810, k=32 F1 0.816, k=64 F1 0.805.
- Consensus all vs tc-only: all-consensus F1 0.814 vs tc-only 0.847; tc-only wins 30.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1920)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1920, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.771
  - best_cycle F1 0.755 (tc-F1 0.769)
  - best_lenpen F1 0.756 (tc-F1 0.766)
- Rerank sweep: best all-F1 0.756 at alpha 0.001 (tc-F1 0.766; tc-rate 1.000).
- Consensus rerank: F1 0.829; tc-F1 0.772; typecheck 0.833.
- Consensus+cycle sweep: best all-F1 0.831 at alpha 0.0 beta 2.0 (tc-F1 0.797).
- Learned reranker (CV, no-cycle): F1 0.815; tc-F1 0.784; typecheck 0.222.
- Oracle headroom: baseline F1 0.771 → oracle 0.939; oracle-tc 0.886.
- Cycle/F1 correlation: pearson avg 0.285, spearman avg 0.079.
- Typecheck signal: tc candidate F1 0.765 vs non-tc 0.767; max tc F1 0.886 vs max non-tc 0.845.
- Consensus k-sweep (all candidates): k=4 F1 0.809, k=8 F1 0.822, k=16 F1 0.829, k=32 F1 0.830, k=64 F1 0.831.
- Consensus all vs tc-only: all-consensus F1 0.829 vs tc-only 0.772; tc-only wins 5.6%.

### High-temp + 128 candidates (temp 0.8, start-index 1940)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1940, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.785
  - best_cycle F1 0.753 (tc-F1 0.785)
  - best_lenpen F1 0.779 (tc-F1 0.799)
- Rerank sweep: best all-F1 0.779 at alpha 0.002 (tc-F1 0.803; tc-rate 1.000).
- Consensus rerank: F1 0.837; tc-F1 0.839; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.849 at alpha 0.0 beta 2.0 (tc-F1 0.853).
- Learned reranker (CV, no-cycle): F1 0.835; tc-F1 0.831; typecheck 0.950.
- Oracle headroom: baseline F1 0.785 → oracle 0.938; oracle-tc 0.928.
- Cycle/F1 correlation: pearson avg 0.311, spearman avg 0.203.
- Typecheck signal: tc candidate F1 0.789 vs non-tc 0.699; max tc F1 0.928 vs max non-tc 0.808.
- Consensus k-sweep (all candidates): k=4 F1 0.807, k=8 F1 0.815, k=16 F1 0.832, k=32 F1 0.838, k=64 F1 0.820.
- Consensus all vs tc-only: all-consensus F1 0.837 vs tc-only 0.839; tc-only wins 15.0%.

### High-temp + 128 candidates (temp 0.8, start-index 1960)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1960, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.764
  - best_cycle F1 0.730 (tc-F1 0.799)
  - best_lenpen F1 0.730 (tc-F1 0.751)
- Rerank sweep: best all-F1 0.730 at alpha 0.0 (tc-F1 0.799; tc-rate 1.000).
- Consensus rerank: F1 0.836; tc-F1 0.861; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.830 at alpha 0.001 beta 3.0 (tc-F1 0.850).
- Learned reranker (CV, no-cycle): F1 0.838; tc-F1 0.857; typecheck 0.947.
- Oracle headroom: baseline F1 0.764 → oracle 0.946; oracle-tc 0.938.
- Cycle/F1 correlation: pearson avg 0.219, spearman avg 0.100.
- Typecheck signal: tc candidate F1 0.795 vs non-tc 0.724; max tc F1 0.938 vs max non-tc 0.854.
- Consensus k-sweep (all candidates): k=4 F1 0.808, k=8 F1 0.818, k=16 F1 0.824, k=32 F1 0.831, k=64 F1 0.833.
- Consensus all vs tc-only: all-consensus F1 0.836 vs tc-only 0.861; tc-only wins 15.8%.

### High-temp + 128 candidates (temp 0.8, start-index 1980)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 1980, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.749
  - best_cycle F1 0.717 (tc-F1 0.736)
  - best_lenpen F1 0.727 (tc-F1 0.733)
- Rerank sweep: best all-F1 0.740 at alpha 0.005 (tc-F1 0.727; tc-rate 1.000).
- Consensus rerank: F1 0.790; tc-F1 0.809; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.778 at alpha 0.005 beta 3.0 (tc-F1 0.817).
- Learned reranker (CV, no-cycle): F1 0.801; tc-F1 0.820; typecheck 0.900.
- Oracle headroom: baseline F1 0.749 → oracle 0.928; oracle-tc 0.910.
- Cycle/F1 correlation: pearson avg 0.177, spearman avg 0.080.
- Typecheck signal: tc candidate F1 0.737 vs non-tc 0.684; max tc F1 0.910 vs max non-tc 0.850.
- Consensus k-sweep (all candidates): k=4 F1 0.768, k=8 F1 0.772, k=16 F1 0.777, k=32 F1 0.772, k=64 F1 0.764.
- Consensus all vs tc-only: all-consensus F1 0.790 vs tc-only 0.809; tc-only wins 20.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2000)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2000, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.761
  - best_cycle F1 0.762 (tc-F1 0.749)
  - best_lenpen F1 0.763 (tc-F1 0.751)
- Rerank sweep: best all-F1 0.763 at alpha 0.001 (tc-F1 0.751; tc-rate 1.000).
- Consensus rerank: F1 0.829; tc-F1 0.785; typecheck 0.737.
- Consensus+cycle sweep: best all-F1 0.837 at alpha 0.0 beta 3.0 (tc-F1 0.795).
- Learned reranker (CV, no-cycle): F1 0.822; tc-F1 0.779; typecheck 0.895.
- Oracle headroom: baseline F1 0.761 → oracle 0.919; oracle-tc 0.854.
- Cycle/F1 correlation: pearson avg 0.434, spearman avg 0.287.
- Typecheck signal: tc candidate F1 0.784 vs non-tc 0.729; max tc F1 0.854 vs max non-tc 0.835.
- Consensus k-sweep (all candidates): k=4 F1 0.809, k=8 F1 0.818, k=16 F1 0.829, k=32 F1 0.824, k=64 F1 0.824.
- Consensus all vs tc-only: all-consensus F1 0.829 vs tc-only 0.785; tc-only wins 15.8%.

### High-temp + 128 candidates (temp 0.8, start-index 2020)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2020, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.843
  - best_cycle F1 0.767 (tc-F1 0.777)
  - best_lenpen F1 0.772 (tc-F1 0.782)
- Rerank sweep: best all-F1 0.772 at alpha 0.001 (tc-F1 0.782; tc-rate 1.000).
- Consensus rerank: F1 0.886; tc-F1 0.877; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.869 at alpha 0.0 beta 3.0 (tc-F1 0.859).
- Learned reranker (CV, no-cycle): F1 0.878; tc-F1 0.882; typecheck 0.947.
- Oracle headroom: baseline F1 0.843 → oracle 0.952; oracle-tc 0.949.
- Cycle/F1 correlation: pearson avg 0.390, spearman avg 0.252.
- Typecheck signal: tc candidate F1 0.818 vs non-tc 0.724; max tc F1 0.949 vs max non-tc 0.891.
- Consensus k-sweep (all candidates): k=4 F1 0.849, k=8 F1 0.867, k=16 F1 0.877, k=32 F1 0.876, k=64 F1 0.868.
- Consensus all vs tc-only: all-consensus F1 0.886 vs tc-only 0.877; tc-only wins 10.5%.

### High-temp + 128 candidates (temp 0.8, start-index 2040)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2040, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.714
  - best_cycle F1 0.704 (tc-F1 0.733)
  - best_lenpen F1 0.704 (tc-F1 0.733)
- Rerank sweep: best all-F1 0.706 at alpha 0.002 (tc-F1 0.737; tc-rate 1.000).
- Consensus rerank: F1 0.811; tc-F1 0.812; typecheck 0.750.
- Consensus+cycle sweep: best all-F1 0.801 at alpha 0.0 beta 3.0 (tc-F1 0.821).
- Learned reranker (CV, no-cycle): F1 0.825; tc-F1 0.835; typecheck 0.950.
- Oracle headroom: baseline F1 0.714 → oracle 0.929; oracle-tc 0.919.
- Cycle/F1 correlation: pearson avg 0.378, spearman avg 0.233.
- Typecheck signal: tc candidate F1 0.753 vs non-tc 0.699; max tc F1 0.919 vs max non-tc 0.862.
- Consensus k-sweep (all candidates): k=4 F1 0.778, k=8 F1 0.789, k=16 F1 0.798, k=32 F1 0.819, k=64 F1 0.820.
- Consensus all vs tc-only: all-consensus F1 0.811 vs tc-only 0.812; tc-only wins 20.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2060)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2060, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.755
  - best_cycle F1 0.768 (tc-F1 0.830)
  - best_lenpen F1 0.760 (tc-F1 0.838)
- Rerank sweep: best all-F1 0.769 at alpha 0.002 (tc-F1 0.851; tc-rate 1.000).
- Consensus rerank: F1 0.864; tc-F1 0.853; typecheck 0.895.
- Consensus+cycle sweep: best all-F1 0.877 at alpha 0.005 beta 3.0 (tc-F1 0.879).
- Learned reranker (CV, no-cycle): F1 0.853; tc-F1 0.869; typecheck 0.947.
- Oracle headroom: baseline F1 0.755 → oracle 0.944; oracle-tc 0.943.
- Cycle/F1 correlation: pearson avg 0.449, spearman avg 0.248.
- Typecheck signal: tc candidate F1 0.803 vs non-tc 0.695; max tc F1 0.943 vs max non-tc 0.852.
- Consensus k-sweep (all candidates): k=4 F1 0.829, k=8 F1 0.849, k=16 F1 0.857, k=32 F1 0.858, k=64 F1 0.842.
- Consensus all vs tc-only: all-consensus F1 0.864 vs tc-only 0.853; tc-only wins 15.8%.

### High-temp + 128 candidates (temp 0.8, start-index 2080)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2080, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.779
  - best_cycle F1 0.785 (tc-F1 0.810)
  - best_lenpen F1 0.777 (tc-F1 0.770)
- Rerank sweep: best all-F1 0.785 at alpha 0.0 (tc-F1 0.810; tc-rate 1.000).
- Consensus rerank: F1 0.840; tc-F1 0.866; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.841 at alpha 0.0 beta 3.0 (tc-F1 0.859).
- Learned reranker (CV, no-cycle): F1 0.852; tc-F1 0.873; typecheck 0.947.
- Oracle headroom: baseline F1 0.779 → oracle 0.945; oracle-tc 0.941.
- Cycle/F1 correlation: pearson avg 0.416, spearman avg 0.262.
- Typecheck signal: tc candidate F1 0.816 vs non-tc 0.709; max tc F1 0.941 vs max non-tc 0.845.
- Consensus k-sweep (all candidates): k=4 F1 0.830, k=8 F1 0.842, k=16 F1 0.843, k=32 F1 0.839, k=64 F1 0.834.
- Consensus all vs tc-only: all-consensus F1 0.840 vs tc-only 0.866; tc-only wins 21.1%.

### High-temp + 128 candidates (temp 0.8, start-index 2100)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2100, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.801
  - best_cycle F1 0.682 (tc-F1 0.722)
  - best_lenpen F1 0.719 (tc-F1 0.749)
- Rerank sweep: best all-F1 0.721 at alpha 0.002 (tc-F1 0.755; tc-rate 1.000).
- Consensus rerank: F1 0.813; tc-F1 0.826; typecheck 0.842.
- Consensus+cycle sweep: best all-F1 0.803 at alpha 0.01 beta 3.0 (tc-F1 0.824).
- Learned reranker (CV, no-cycle): F1 0.815; tc-F1 0.815; typecheck 1.000.
- Learned reranker (CV, full features): F1 0.817; tc-F1 0.817; typecheck 1.000.
- Oracle headroom: baseline F1 0.801 → oracle 0.933; oracle-tc 0.908.
- Cycle/F1 correlation: pearson avg 0.275, spearman avg 0.141.
- Typecheck signal: tc candidate F1 0.775 vs non-tc 0.709; max tc F1 0.908 vs max non-tc 0.884.
- Consensus k-sweep (all candidates): k=4 F1 0.800, k=8 F1 0.808, k=16 F1 0.818, k=32 F1 0.817, k=64 F1 0.804.
- Consensus all vs tc-only: all-consensus F1 0.813 vs tc-only 0.826; tc-only wins 15.8%.

### High-temp + 128 candidates (temp 0.8, start-index 2120)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2120, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.787
  - best_cycle F1 0.713 (tc-F1 0.796)
  - best_lenpen F1 0.715 (tc-F1 0.789)
- Rerank sweep: best all-F1 0.715 at alpha 0.001 (tc-F1 0.789; tc-rate 0.947).
- Consensus rerank: F1 0.818; tc-F1 0.815; typecheck 0.789.
- Consensus+cycle sweep: best all-F1 0.815 at alpha 0.0 beta 2.0 (tc-F1 0.828).
- Learned reranker (CV, no-cycle): F1 0.827; tc-F1 0.827; typecheck 0.947.
- Learned reranker (CV, full features): F1 0.816; tc-F1 0.816; typecheck 0.947.
- Oracle headroom: baseline F1 0.787 → oracle 0.925; oracle-tc 0.921.
- Cycle/F1 correlation: pearson avg 0.350, spearman avg 0.148.
- Typecheck signal: tc candidate F1 0.793 vs non-tc 0.662; max tc F1 0.936 vs max non-tc 0.855.
- Consensus k-sweep (all candidates): k=4 F1 0.798, k=8 F1 0.812, k=16 F1 0.820, k=32 F1 0.815, k=64 F1 0.806.
- Consensus all vs tc-only: all-consensus F1 0.818 vs tc-only 0.815; tc-only wins 16.7%.

### High-temp + 128 candidates (temp 0.8, start-index 2140)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2140, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.748
  - best_cycle F1 0.781 (tc-F1 0.788)
  - best_lenpen F1 0.768 (tc-F1 0.789)
- Rerank sweep: best all-F1 0.781 at alpha 0.0 (tc-F1 0.788; tc-rate 0.947).
- Consensus rerank: F1 0.803; tc-F1 0.799; typecheck 0.947.
- Consensus+cycle sweep: best all-F1 0.819 at alpha 0.002 beta 2.0 (tc-F1 0.816).
- Learned reranker (CV, no-cycle): F1 0.805; tc-F1 0.803; typecheck 0.842.
- Learned reranker (CV, full features): F1 0.799; tc-F1 0.798; typecheck 0.842.
- Oracle headroom: baseline F1 0.748 → oracle 0.907; oracle-tc 0.881.
- Cycle/F1 correlation: pearson avg 0.326, spearman avg 0.178.
- Typecheck signal: tc candidate F1 0.762 vs non-tc 0.700; max tc F1 0.884 vs max non-tc 0.834.
- Consensus k-sweep (all candidates): k=4 F1 0.777, k=8 F1 0.789, k=16 F1 0.799, k=32 F1 0.797, k=64 F1 0.794.
- Consensus all vs tc-only: all-consensus F1 0.803 vs tc-only 0.799; tc-only wins 5.6%.

### High-temp + 128 candidates (temp 0.8, start-index 2160)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2160, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.754
  - best_cycle F1 0.712 (tc-F1 0.770)
  - best_lenpen F1 0.712 (tc-F1 0.777)
- Rerank sweep: best all-F1 0.712 at alpha 0.001 (tc-F1 0.777; tc-rate 1.000).
- Consensus rerank: F1 0.819; tc-F1 0.824; typecheck 0.950.
- Consensus+cycle sweep: best all-F1 0.822 at alpha 0.002 beta 2.0 (tc-F1 0.829).
- Learned reranker (CV, no-cycle): F1 0.823; tc-F1 0.822; typecheck 0.800.
- Learned reranker (CV, full features): F1 0.809; tc-F1 0.816; typecheck 0.850.
- Oracle headroom: baseline F1 0.754 → oracle 0.918; oracle-tc 0.907.
- Cycle/F1 correlation: pearson avg 0.322, spearman avg 0.226.
- Typecheck signal: tc candidate F1 0.751 vs non-tc 0.708; max tc F1 0.907 vs max non-tc 0.865.
- Consensus k-sweep (all candidates): k=4 F1 0.790, k=8 F1 0.796, k=16 F1 0.805, k=32 F1 0.813, k=64 F1 0.803.
- Consensus all vs tc-only: all-consensus F1 0.819 vs tc-only 0.824; tc-only wins 15.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2180)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2180, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.758
  - best_cycle F1 0.719 (tc-F1 0.788)
  - best_lenpen F1 0.701 (tc-F1 0.793)
- Rerank sweep: best all-F1 0.719 at alpha 0.0 (tc-F1 0.788; tc-rate 1.000).
- Consensus rerank: F1 0.857; tc-F1 0.860; typecheck 0.850.
- Consensus+cycle sweep: best all-F1 0.835 at alpha 0.0 beta 3.0 (tc-F1 0.848).
- Learned reranker (CV, no-cycle): F1 0.866; tc-F1 0.866; typecheck 1.000.
- Learned reranker (CV, full features): F1 0.866; tc-F1 0.866; typecheck 1.000.
- Oracle headroom: baseline F1 0.758 → oracle 0.939; oracle-tc 0.938.
- Cycle/F1 correlation: pearson avg 0.197, spearman avg 0.024.
- Typecheck signal: tc candidate F1 0.792 vs non-tc 0.725; max tc F1 0.938 vs max non-tc 0.870.
- Consensus k-sweep (all candidates): k=4 F1 0.826, k=8 F1 0.842, k=16 F1 0.851, k=32 F1 0.847, k=64 F1 0.835.
- Consensus all vs tc-only: all-consensus F1 0.857 vs tc-only 0.860; tc-only wins 15.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2200)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2200, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.784
  - best_cycle F1 0.714 (tc-F1 0.713)
  - best_lenpen F1 0.710 (tc-F1 0.709)
- Rerank sweep: best all-F1 0.714 at alpha 0.0 (tc-F1 0.713; tc-rate 1.000).
- Consensus rerank: F1 0.841; tc-F1 0.828; typecheck 0.800.
- Consensus+cycle sweep: best all-F1 0.831 at alpha 0.001 beta 2.0 (tc-F1 0.833).
- Learned reranker (CV, no-cycle): F1 0.842; tc-F1 0.837; typecheck 0.800.
- Learned reranker (CV, full features): F1 0.844; tc-F1 0.837; typecheck 0.800.
- Oracle headroom: baseline F1 0.784 → oracle 0.927; oracle-tc 0.917.
- Cycle/F1 correlation: pearson avg 0.242, spearman avg 0.104.
- Typecheck signal: tc candidate F1 0.772 vs non-tc 0.750; max tc F1 0.917 vs max non-tc 0.845.
- Consensus k-sweep (all candidates): k=4 F1 0.812, k=8 F1 0.824, k=16 F1 0.831, k=32 F1 0.833, k=64 F1 0.824.
- Consensus all vs tc-only: all-consensus F1 0.841 vs tc-only 0.789; tc-only wins 10.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2220)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2220, 20 requested (19 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.761
  - best_cycle F1 0.708 (tc-F1 0.713)
  - best_lenpen F1 0.695 (tc-F1 0.715)
- Rerank sweep: best all-F1 0.745 at alpha 0.005 (tc-F1 0.691; tc-rate 1.000).
- Consensus rerank: F1 0.787; tc-F1 0.780; typecheck 0.684.
- Consensus+cycle sweep: best all-F1 0.774 at alpha 0.002 beta 3.0 (tc-F1 0.764).
- Learned reranker (CV, no-cycle): F1 0.792; tc-F1 0.781; typecheck 0.947.
- Learned reranker (CV, full features): F1 0.795; tc-F1 0.785; typecheck 0.947.
- Oracle headroom: baseline F1 0.761 → oracle 0.910; oracle-tc 0.862.
- Cycle/F1 correlation: pearson avg 0.199, spearman avg 0.121.
- Typecheck signal: tc candidate F1 0.763 vs non-tc 0.699; max tc F1 0.862 vs max non-tc 0.847.
- Consensus k-sweep (all candidates): k=4 F1 0.768, k=8 F1 0.781, k=16 F1 0.788, k=32 F1 0.789, k=64 F1 0.782.
- Consensus all vs tc-only: all-consensus F1 0.787 vs tc-only 0.780; tc-only wins 31.6%.

### High-temp + 128 candidates (temp 0.8, start-index 2240)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2240, 20 requested (20 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.835
  - best_cycle F1 0.708 (tc-F1 0.740)
  - best_lenpen F1 0.705 (tc-F1 0.738)
- Rerank sweep: best all-F1 0.720 at alpha 0.002 (tc-F1 0.754; tc-rate 1.000).
- Consensus rerank: F1 0.878; tc-F1 0.882; typecheck 0.900.
- Consensus+cycle sweep: best all-F1 0.868 at alpha 0.0 beta 3.0 (tc-F1 0.868).
- Learned reranker (CV, no-cycle): F1 0.873; tc-F1 0.874; typecheck 0.900.
- Learned reranker (CV, full features): F1 0.877; tc-F1 0.878; typecheck 0.900.
- Oracle headroom: baseline F1 0.835 → oracle 0.932; oracle-tc 0.932.
- Cycle/F1 correlation: pearson avg 0.276, spearman avg 0.117.
- Typecheck signal: tc candidate F1 0.817 vs non-tc 0.773; max tc F1 0.932 vs max non-tc 0.873.
- Consensus k-sweep (all candidates): k=4 F1 0.853, k=8 F1 0.865, k=16 F1 0.870, k=32 F1 0.862, k=64 F1 0.860.
- Consensus all vs tc-only: all-consensus F1 0.878 vs tc-only 0.882; tc-only wins 15.0%.

### High-temp + 128 candidates (temp 0.8, start-index 2260)

- Run: num_candidates 128, temperature 0.8, top-p 0.95, length_penalty 0.001, start_index 2260, 20 requested (18 unique, 3-GPU sharded), CPU-rescored (local snapshot path).
- Token-level statement similarity:
  - baseline F1 0.752
  - best_cycle F1 0.677 (tc-F1 0.750)
  - best_lenpen F1 0.677 (tc-F1 0.742)
- Rerank sweep: best all-F1 0.678 at alpha 0.002 (tc-F1 0.753; tc-rate 1.000).
- Consensus rerank: F1 0.793; tc-F1 0.790; typecheck 0.889.
- Consensus+cycle sweep: best all-F1 0.787 at alpha 0.01 beta 3.0 (tc-F1 0.803).
- Learned reranker (CV, no-cycle): F1 0.765; tc-F1 0.792; typecheck 0.556.
- Learned reranker (CV, full features): F1 0.767; tc-F1 0.793; typecheck 0.556.
- Oracle headroom: baseline F1 0.752 → oracle 0.895; oracle-tc 0.865.
- Cycle/F1 correlation: pearson avg 0.365, spearman avg 0.215.
- Typecheck signal: tc candidate F1 0.736 vs non-tc 0.714; max tc F1 0.865 vs max non-tc 0.834.
- Consensus k-sweep (all candidates): k=4 F1 0.764, k=8 F1 0.778, k=16 F1 0.782, k=32 F1 0.781, k=64 F1 0.772.
- Consensus all vs tc-only: all-consensus F1 0.793 vs tc-only 0.790; tc-only wins 11.1%.

### Consensus vs subset size (64-candidate slice)

- New script: `analysis_consensus_k_sweep.py` (consensus F1 vs random subset size).
- All candidates: k=4 F1 0.821, k=8 F1 0.831, k=16 F1 0.836, k=32 F1 0.826, k=64 F1 0.901 (k=64 only 2 records).
- Typechecked-only: k=4 F1 0.828, k=8 F1 0.826, k=16 F1 0.835, k=32 F1 0.808.
- 32-candidate slice: all F1 grows with k (k=4 0.797 → k=32 0.841); tc-only peaks at k=16 (0.885, fewer records).
- 16-candidate slice: all F1 declines with k (k=4 0.779 → k=16 0.761); tc-only peaks at k=8 (0.797).
- 128-candidate slice (start-index 1580): all F1 peaks at k=32 (0.874) and drops at k=128 (0.810); tc-only peaks at k=16 (0.884).
- 128-candidate slice (start-index 1600): all F1 peaks at k=32 (0.875); tc-only peaks at k=64 (0.897, fewer records); k=128 unavailable.
- 128-candidate slice (start-index 1620): all F1 peaks at k=32 (0.787); tc-only peaks at k=32 (0.819); k=128 available for 2 records (F1 0.777).
- 128-candidate slice (start-index 1640): all F1 peaks at k=32 (0.863); tc-only peaks at k=32 (0.870).
- 128-candidate slice (start-index 1660): all F1 peaks at k=64 (0.841); tc-only peaks at k=32 (0.864).
- 128-candidate slice (start-index 1680): all F1 peaks at k=32 (0.836); tc-only peaks at k=32 (0.848).
- 128-candidate slice (start-index 1700): all F1 peaks at k=32 (0.802); tc-only peaks at k=64 (0.858, fewer records).
- 128-candidate slice (start-index 1720): all F1 peaks at k=32 (0.843); tc-only peaks at k=16 (0.852).
- 128-candidate slice (start-index 1740): all F1 peaks at k=16 (0.832); tc-only peaks at k=16 (0.844).
- 128-candidate slice (start-index 1760): all F1 peaks at k=32 (0.828); tc-only peaks at k=64 (0.826, fewer records).
- 128-candidate slice (start-index 1780): all F1 peaks at k=64 (0.864); tc-only peaks at k=16 (0.873, fewer records).
- 128-candidate slice (start-index 1800): all F1 peaks at k=32 (0.863); tc-only peaks at k=64 (0.869, fewer records).
- 128-candidate slice (start-index 1820): all F1 peaks at k=16 (0.847); tc-only peaks at k=32 (0.844, fewer records).
- 128-candidate slice (start-index 1840): all F1 peaks at k=32 (0.862); tc-only peaks at k=64 (0.890, fewer records).
- 128-candidate slice (start-index 1860): all F1 peaks at k=64 (0.828); tc-only peaks at k=32 (0.830, fewer records).
- 128-candidate slice (start-index 1880): all F1 peaks at k=32 (0.853); tc-only peaks at k=64 (0.879, fewer records).
- 128-candidate slice (start-index 1900): all F1 peaks at k=32 (0.816); tc-only peaks at k=64 (0.832, fewer records).
- 128-candidate slice (start-index 1920): all F1 peaks at k=64 (0.831); tc-only peaks at k=32 (0.821).
- 128-candidate slice (start-index 1940): all F1 peaks at k=32 (0.838); tc-only peaks at k=32 (0.856).
- 128-candidate slice (start-index 1960): all F1 peaks at k=64 (0.833); tc-only peaks at k=16 (0.860).
- 128-candidate slice (start-index 1980): all F1 peaks at k=16 (0.777); tc-only peaks at k=8 (0.801).
- 128-candidate slice (start-index 2000): all F1 peaks at k=16 (0.829); tc-only peaks at k=64 (0.846).
- 128-candidate slice (start-index 2020): all F1 peaks at k=16 (0.877); tc-only peaks at k=32 (0.894).
- 128-candidate slice (start-index 2040): all F1 peaks at k=64 (0.820); tc-only peaks at k=32 (0.818).
- 128-candidate slice (start-index 2060): all F1 peaks at k=32 (0.858); tc-only peaks at k=32 (0.876).
- 128-candidate slice (start-index 2080): all F1 peaks at k=16 (0.843); tc-only peaks at k=16 (0.875).
- 128-candidate slice (start-index 2100): all F1 peaks at k=16 (0.818); tc-only peaks at k=64 (0.862, fewer records).
- 128-candidate slice (start-index 2120): all F1 peaks at k=16 (0.820); tc-only peaks at k=16 (0.852).
- 128-candidate slice (start-index 2140): all F1 peaks at k=16 (0.799); tc-only peaks at k=32 (0.809).
- 128-candidate slice (start-index 2160): all F1 peaks at k=32 (0.813); tc-only peaks at k=32 (0.811).
- 128-candidate slice (start-index 2180): all F1 peaks at k=16 (0.851); tc-only peaks at k=16 (0.853).
- 128-candidate slice (start-index 2200): all F1 peaks at k=32 (0.833); tc-only peaks at k=16 (0.831).
- 128-candidate slice (start-index 2220): all F1 peaks at k=32 (0.789); tc-only peaks at k=64 (0.844, fewer records).
- 128-candidate slice (start-index 2240): all F1 peaks at k=16 (0.870); tc-only peaks at k=64 (0.891, fewer records).
- 128-candidate slice (start-index 2260): all F1 peaks at k=16 (0.782); tc-only peaks at k=64 (0.810, fewer records).
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640): k=32 all F1 0.850 (trial-weighted), full consensus 0.845 (79 records).
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680, 86 unique): k=32 all F1 0.840, full consensus 0.839.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700, 96 unique): k=32 all F1 0.835, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720, 99 unique): k=32 all F1 0.834, full consensus 0.835.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740, 105 unique): k=32 all F1 0.833, full consensus 0.834.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760, 108 unique): k=32 all F1 0.835, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780, 113 unique): k=32 all F1 0.838, full consensus 0.839.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800, 116 unique): k=32 all F1 0.837, full consensus 0.839.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820, 121 unique): k=32 all F1 0.834, full consensus 0.836.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840, 126 unique): k=32 all F1 0.838, full consensus 0.838.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860, 129 unique): k=32 all F1 0.836, full consensus 0.838.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880, 134 unique): k=32 all F1 0.833, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900, 142 unique): k=32 all F1 0.834, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920, 144 unique): k=32 all F1 0.835, full consensus 0.839.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940, 147 unique): k=32 all F1 0.835, full consensus 0.838.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960, 150 unique): k=32 all F1 0.834, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980, 151 unique): k=32 all F1 0.831, full consensus 0.834.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000, 152 unique): k=32 all F1 0.831, full consensus 0.833.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020, 156 unique): k=32 all F1 0.834, full consensus 0.835.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040, 158 unique): k=32 all F1 0.833, full consensus 0.836.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060, 159 unique): k=32 all F1 0.833, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080, 161 unique): k=32 all F1 0.834, full consensus 0.836.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100, 163 unique): k=32 all F1 0.832, full consensus 0.834.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120, 182 unique): k=32 all F1 0.830, full consensus 0.833.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140, 201 records): k=32 all F1 0.827, full consensus 0.830.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160, 221 records): k=32 all F1 0.826, full consensus 0.829.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180, 241 records): k=32 all F1 0.827, full consensus 0.831.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200, 261 records): k=32 all F1 0.827, full consensus 0.832.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220, 630 records): k=32 all F1 0.833, full consensus 0.835.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240, 650 records): k=32 all F1 0.834, full consensus 0.837.
- 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240 + 2260, 668 records): k=32 all F1 0.832, full consensus 0.835.

### Cycle top-k consensus (128-candidate slices)

- New script: `analysis_cycle_consensus_k_sweep.py` (consensus after selecting top-k by cycle or length-penalized score).
- Start-index 1580: best all-F1 0.868 at k=64 (lenpen top-k), below full consensus 0.875; tc-only best 0.889 at k=32.
- Start-index 1600: best all-F1 0.876 at k=32 (lenpen top-k), slightly above full consensus 0.872; tc-only best 0.900 at k=64 (few records).
- Start-index 1620: best all-F1 0.780 at k=8 (lenpen top-k), slightly above full consensus 0.774; tc-only best 0.816 at k=32 (cycle top-k).
- Start-index 1640: best all-F1 0.858 at k=32 (lenpen top-k), just below full consensus 0.861; tc-only best 0.867 at k=16 (cycle top-k).
- Start-index 1660: best all-F1 0.833 at k=32 (lenpen top-k), below full consensus 0.839; tc-only best 0.855 at k=16/32.
- Start-index 1680: best all-F1 0.844 at k=32 (cycle top-k), above full consensus 0.836; tc-only best 0.860 at k=32.
- Start-index 1700: best all-F1 0.789 at k=64 (cycle top-k), below full consensus 0.799; tc-only best 0.864 at k=64 (few records).
- Start-index 1720: best all-F1 0.849 at k=32 (cycle top-k), below full consensus 0.852; tc-only best 0.858 at k=16.
- Start-index 1740: best all-F1 0.840 at k=32 (lenpen top-k), above full consensus 0.825; tc-only best 0.845 at k=32 (cycle top-k).
- Start-index 1760: best all-F1 0.821 at k=32 (cycle top-k), below full consensus 0.843; tc-only best 0.820 at k=8 (cycle top-k).
- Start-index 1780: best all-F1 0.867 at k=64 (cycle top-k), below full consensus 0.872; tc-only best 0.866 at k=8 (cycle top-k).
- Start-index 1800: best all-F1 0.881 at k=32 (lenpen top-k), above full consensus 0.860; tc-only best 0.889 at k=32 (lenpen top-k).
- Start-index 1820: best all-F1 0.839 at k=64 (lenpen top-k), below full consensus 0.843; tc-only best 0.845 at k=32 (lenpen top-k).
- Start-index 1840: best all-F1 0.861 at k=64 (lenpen top-k), above full consensus 0.853; tc-only best 0.888 at k=64 (cycle top-k).
- Start-index 1860: best all-F1 0.829 at k=32 (cycle top-k), above full consensus 0.823; tc-only best 0.829 at k=32 (cycle top-k).
- Start-index 1880: best all-F1 0.855 at k=4 (lenpen top-k), below full consensus 0.865; tc-only best 0.876 at k=64 (cycle/lenpen top-k).
- Start-index 1900: best all-F1 0.828 at k=16 (cycle top-k), above full consensus 0.814; tc-only best 0.844 at k=8 (lenpen top-k).
- Start-index 1920: best all-F1 0.827 at k=32 (cycle top-k), below full consensus 0.829; tc-only best 0.828 at k=16 (cycle top-k).
- Start-index 1940: best all-F1 0.838 at k=32 (cycle top-k), above full consensus 0.837; tc-only best 0.865 at k=16 (cycle top-k).
- Start-index 1960: best all-F1 0.835 at k=32/64 (cycle top-k), below full consensus 0.836; tc-only best 0.853 at k=64 (cycle/lenpen top-k).
- Start-index 1980: best all-F1 0.779 at k=16 (cycle top-k), below full consensus 0.790; tc-only best 0.800 at k=8 (lenpen top-k).
- Start-index 2000: best all-F1 0.830 at k=64 (lenpen top-k), above full consensus 0.829; tc-only best 0.851 at k=16 (cycle top-k).
- Start-index 2020: best all-F1 0.898 at k=32 (cycle top-k), above full consensus 0.886; tc-only best 0.897 at k=32 (cycle top-k).
- Start-index 2040: best all-F1 0.830 at k=64 (cycle top-k), above full consensus 0.811; tc-only best 0.805 at k=32 (cycle top-k).
- Start-index 2060: best all-F1 0.867 at k=16 (cycle top-k), above full consensus 0.864; tc-only best 0.880 at k=32 (lenpen top-k).
- Start-index 2080: best all-F1 0.854 at k=16 (lenpen top-k), above full consensus 0.840; tc-only best 0.879 at k=16 (lenpen top-k).
- Start-index 2100: best all-F1 0.805 at k=32 (cycle top-k), below full consensus 0.813; tc-only best 0.855 at k=64 (cycle/lenpen top-k).
- Start-index 2120: best all-F1 0.825 at k=32 (cycle top-k), above full consensus 0.818; tc-only best 0.846 at k=16 (cycle top-k).
- Start-index 2140: best all-F1 0.811 at k=16 (cycle top-k), above full consensus 0.803; tc-only best 0.803 at k=4 (cycle top-k).
- Start-index 2160: best all-F1 0.823 at k=32 (cycle top-k), above full consensus 0.819; tc-only best 0.823 at k=16 (cycle top-k).
- Start-index 2180: best all-F1 0.847 at k=16 (cycle top-k), below full consensus 0.857; tc-only best 0.845 at k=64 (cycle top-k, fewer records).
- Start-index 2200: best all-F1 0.828 at k=32 (lenpen top-k), below full consensus 0.841; tc-only best 0.834 at k=16 (cycle top-k).
- Start-index 2220: best all-F1 0.793 at k=64 (cycle top-k), above full consensus 0.787; tc-only best 0.846 at k=64 (cycle top-k, fewer records).
- Start-index 2240: best all-F1 0.850 at k=32 (cycle top-k), below full consensus 0.878; tc-only best 0.883 at k=64 (cycle top-k, fewer records).
- Start-index 2260: best all-F1 0.793 at k=32 (lenpen top-k), below full consensus 0.793; tc-only best 0.823 at k=64 (cycle/lenpen top-k, fewer records).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640): best all-F1 0.844 at k=32 (lenpen top-k), below full consensus 0.845 (79 records).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940): best all-F1 0.830 at k=32 (cycle top-k), below full consensus 0.838 (147 records); tc-only best 0.851 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960): best all-F1 0.829 at k=32 (cycle top-k), below full consensus 0.837 (150 records); tc-only best 0.851 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980): best all-F1 0.826 at k=32 (cycle top-k), below full consensus 0.834 (151 records); tc-only best 0.847 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000): best all-F1 0.825 at k=32 (cycle top-k), below full consensus 0.833 (152 records); tc-only best 0.847 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020): best all-F1 0.828 at k=32 (cycle top-k), below full consensus 0.835 (156 records); tc-only best 0.849 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040): best all-F1 0.827 at k=64 (cycle top-k), below full consensus 0.836 (158 records); tc-only best 0.847 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060): best all-F1 0.827 at k=64 (cycle top-k), below full consensus 0.837 (159 records); tc-only best 0.847 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080): best all-F1 0.829 at k=64 (cycle top-k), below full consensus 0.836 (161 records); tc-only best 0.848 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100): best all-F1 0.826 at k=64 (cycle top-k), below full consensus 0.834 (163 records); tc-only best 0.848 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120): best all-F1 0.825 at k=32 (cycle top-k), below full consensus 0.833 (182 records); tc-only best 0.847 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140): best all-F1 0.821 at k=32/64 (cycle top-k), below full consensus 0.830 (201 records); tc-only best 0.842 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160): best all-F1 0.821 at k=32 (cycle top-k), below full consensus 0.829 (221 records); tc-only best 0.839 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180): best all-F1 0.822 at k=32 (cycle top-k), below full consensus 0.831 (241 records); tc-only best 0.839 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200): best all-F1 0.823 at k=32 (cycle top-k), below full consensus 0.832 (261 records); tc-only best 0.838 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220): best all-F1 0.832 at k=32 (cycle top-k), below full consensus 0.835 (630 records); tc-only best 0.841 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240): best all-F1 0.832 at k=32 (cycle top-k), below full consensus 0.837 (650 records); tc-only best 0.841 at k=32 (cycle top-k).
- Combined 128-candidate slices (1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240 + 2260): best all-F1 0.831 at k=32 (cycle top-k), below full consensus 0.835 (668 records); tc-only best 0.839 at k=32 (cycle top-k).

### TC-gated consensus (128-candidate slices)

- New script: `analysis_consensus_tc_gate_sweep.py` (gate between all-consensus and tc-only consensus).
- New script: `analysis_consensus_tc_gate_learned.py` (learn a gate from record-level features).
- Extended gating features: pairwise agreement, pairwise_tc agreement, consensus margin.
- Combined 128-candidate slices (1580+1600+1620+1640, 79 records):
  - all-consensus F1 0.845; tc-only consensus 0.846.
  - gating by tc_rate >= 0.4 yields F1 0.855; tc_rate >= 0.333 (from 64-cand tuning) yields F1 0.855 (slightly higher).
  - gating by tc_count >= 16 yields F1 0.852; tc_count >= 8 yields F1 0.852.
- Combined 128-candidate slices (1580+1600+1620+1640+1660, 98 records):
  - all-consensus F1 0.844; tc-only consensus 0.850.
  - best tc_rate gate at 0.168 yields F1 0.856; tc_rate 0.333 yields F1 0.854.
  - tc_count >= 17 yields F1 0.853.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680, 115 records):
  - all-consensus F1 0.843; tc-only consensus 0.848.
  - best tc_rate gate at 0.194 yields F1 0.854; tc_rate 0.333 yields F1 0.850.
  - tc_count >= 17 yields F1 0.849.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700, 134 records):
  - all-consensus F1 0.837; tc-only consensus 0.842.
  - best tc_rate gate at 0.194 yields F1 0.846; tc_rate 0.333 yields F1 0.843.
  - tc_count gating never beats tc-only (best is tc-only F1 0.842).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720, 152 records):
  - all-consensus F1 0.838; tc-only consensus 0.844.
  - best tc_rate gate at 0.194 yields F1 0.847; tc_rate 0.333 yields F1 0.844.
  - tc_count gating never beats tc-only (best is tc-only F1 0.844).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740, 170 records):
  - all-consensus F1 0.837; tc-only consensus 0.842.
  - best tc_rate gate at 0.194 yields F1 0.846; tc_rate 0.333 yields F1 0.843.
  - tc_count gating never beats tc-only (best is tc-only F1 0.842).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780, 113 records):
  - all-consensus F1 0.839; tc-only consensus 0.844.
  - best tc_rate gate at 0.351 yields F1 0.847; tc_rate 0.333 yields F1 0.847.
  - tc_count >= 16 yields F1 0.846.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800, 116 records):
  - all-consensus F1 0.839; tc-only consensus 0.844.
  - best tc_rate gate at 0.168 yields F1 0.849; tc_rate 0.333 yields F1 0.848.
  - tc_count >= 17 yields F1 0.846.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820, 121 records):
  - all-consensus F1 0.836; tc-only consensus 0.841.
  - best tc_rate gate at 0.168 yields F1 0.846; tc_rate 0.333 yields F1 0.844.
  - tc_count >= 17 yields F1 0.843.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840, 126 records):
  - all-consensus F1 0.838; tc-only consensus 0.843.
  - best tc_rate gate at 0.168 yields F1 0.847; tc_rate 0.333 yields F1 0.846.
  - tc_count >= 17 yields F1 0.844.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860, 129 records):
  - all-consensus F1 0.838; tc-only consensus 0.843.
  - best tc_rate gate at 0.194 yields F1 0.847; tc_rate 0.333 yields F1 0.846.
  - tc_count >= 17 yields F1 0.844.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880, 134 records):
  - all-consensus F1 0.837; tc-only consensus 0.842.
  - best tc_rate gate at 0.373 yields F1 0.844; tc_rate 0.333 yields F1 0.844.
  - tc_count >= 16 yields F1 0.842.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900, 142 records):
  - all-consensus F1 0.837; tc-only consensus 0.845.
  - best tc_rate gate at 0.0 yields F1 0.845; tc_rate 0.333 yields F1 0.844.
  - tc_count >= 16 yields F1 0.843.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920, 144 records):
  - all-consensus F1 0.839; tc-only consensus 0.845.
  - best tc_rate gate at 0.0 yields F1 0.845; tc_rate 0.333 yields F1 0.845.
  - tc_count >= 16 yields F1 0.843.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940, 147 records):
  - all-consensus F1 0.838; tc-only consensus 0.843.
  - best tc_rate gate at 0.367 yields F1 0.843; tc_rate 0.333 yields F1 0.844.
  - tc_count >= 16 yields F1 0.842.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960, 150 records):
  - all-consensus F1 0.837; tc-only consensus 0.842.
  - best tc_rate gate at 0.367 yields F1 0.843; tc_rate 0.333 yields F1 0.843.
  - tc_count gating picks tc-only (F1 0.842).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980, 151 records):
  - all-consensus F1 0.834; tc-only consensus 0.839.
  - best tc_rate gate at 0.367 yields F1 0.840; tc_rate 0.333 yields F1 0.840.
  - tc_count gating picks tc-only (F1 0.839).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000, 152 records):
  - all-consensus F1 0.833; tc-only consensus 0.839.
  - best tc_rate gate at 0.351 yields F1 0.839; tc_rate 0.333 yields F1 0.839.
  - tc_count gating picks tc-only (F1 0.839).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020, 156 records):
  - all-consensus F1 0.835; tc-only consensus 0.840.
  - best tc_rate gate at 0.367 yields F1 0.841; tc_rate 0.333 yields F1 0.841.
  - tc_count gating picks tc-only (F1 0.840).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040, 158 records):
  - all-consensus F1 0.836; tc-only consensus 0.839.
  - best tc_rate gate at 0.351 yields F1 0.840; tc_rate 0.333 yields F1 0.840.
  - tc_count gating picks tc-only (F1 0.839).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060, 159 records):
  - all-consensus F1 0.837; tc-only consensus 0.840.
  - best tc_rate gate at 0.367 yields F1 0.840; tc_rate 0.333 yields F1 0.840.
  - tc_count gating picks tc-only (F1 0.840).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080, 161 records):
  - all-consensus F1 0.836; tc-only consensus 0.839.
  - best tc_rate gate at 0.351 yields F1 0.840; tc_rate 0.333 yields F1 0.840.
  - tc_count gating picks tc-only (F1 0.839).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100, 163 records):
  - all-consensus F1 0.834; tc-only consensus 0.839.
  - best tc_rate gate at 0.0 yields F1 0.839; tc_rate 0.333 yields F1 0.838.
  - tc_count gating picks tc-only (F1 0.839).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120, 182 records):
  - all-consensus F1 0.833; tc-only consensus 0.836.
  - best tc_rate gate at 0.0 yields F1 0.836; tc_rate 0.333 yields F1 0.836.
  - tc_count gating picks tc-only (F1 0.836).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140, 201 records):
  - all-consensus F1 0.830; tc-only consensus 0.833.
  - best tc_rate gate at 0.149 yields F1 0.833; tc_rate 0.333 yields F1 0.832.
  - tc_count gating picks tc-only (F1 0.833).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160, 221 records):
  - all-consensus F1 0.829; tc-only consensus 0.832.
  - best tc_rate gate at 0.172 yields F1 0.833; tc_rate 0.333 yields F1 0.831.
  - tc_count gating picks tc-only (F1 0.832).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160+2180, 241 records):
  - all-consensus F1 0.831; tc-only consensus 0.834.
  - best tc_rate gate at 0.194 yields F1 0.835; tc_rate 0.333 yields F1 0.834.
  - tc_count gating picks tc-only (F1 0.834).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160+2180+2200, 261 records):
  - all-consensus F1 0.832; tc-only consensus 0.834.
  - best tc_rate gate at 0.179 yields F1 0.835; tc_rate 0.333 yields F1 0.834.
  - tc_count gating picks tc-only (F1 0.834).
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160+2180+2200+2220, 630 records):
  - all-consensus F1 0.835; tc-only consensus 0.835.
  - best tc_rate gate at 0.220 yields F1 0.838; tc_rate 0.333 yields F1 0.837.
  - best tc_count gate at 19 yields F1 0.837.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160+2180+2200+2220+2240, 650 records):
  - all-consensus F1 0.837; tc-only consensus 0.836.
  - best tc_rate gate at 0.233 yields F1 0.840; tc_rate 0.333 yields F1 0.838.
  - best tc_count gate at 19 yields F1 0.838.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760+1780+1800+1820+1840+1860+1880+1900+1920+1940+1960+1980+2000+2020+2040+2060+2080+2100+2120+2140+2160+2180+2200+2220+2240+2260, 668 records):
  - all-consensus F1 0.835; tc-only consensus 0.835.
  - best tc_rate gate at 0.233 yields F1 0.838; tc_rate 0.333 yields F1 0.837.
  - best tc_count gate at 19 yields F1 0.837.
- Combined 128-candidate slices (1580+1600+1620+1640+1660+1680+1700+1720+1740+1760, 108 records):
  - all-consensus F1 0.837; tc-only consensus 0.841.
  - best tc_rate gate at 0.351 yields F1 0.845; tc_rate 0.333 yields F1 0.845.
  - tc_count >= 16 yields F1 0.843.
- Per-slice tc_rate>=0.4 gate: start1580 0.873, start1600 0.884, start1620 0.791, start1640 0.874.
- Start1660: tc-only consensus 0.865 beats all-consensus 0.839; best tc_rate gate picks tc-only (F1 0.865).
- Start1680: tc-only consensus 0.838 beats all-consensus 0.836; best tc_rate gate picks tc-only (F1 0.838).
- Start1700: tc-only consensus 0.804 beats all-consensus 0.799; best tc_rate gate picks tc-only (F1 0.804).
- Start1720: tc-only consensus 0.858 beats all-consensus 0.852; best tc_rate gate picks tc-only (F1 0.858).
- Start1740: tc_rate >= 0.426 gate yields F1 0.837 (all-consensus 0.825; tc-only 0.793).
- Start1760: tc-only consensus 0.824 below all-consensus 0.843; best tc_rate gate picks all-consensus (F1 0.843).
- Start1780: tc-only consensus 0.876 beats all-consensus 0.872; best tc_rate gate picks tc-only (F1 0.876).
- Start1800: tc-only consensus 0.872 beats all-consensus 0.860; best tc_rate gate picks tc-only (F1 0.872).
- Start1820: tc-only consensus 0.793 below all-consensus 0.843; best tc_rate gate yields F1 0.848.
- Start1840: tc-only consensus 0.846 below all-consensus 0.853; best tc_count gate yields F1 0.857.
- Start1860: tc-only consensus 0.821 below all-consensus 0.823; best tc_count gate yields F1 0.825.
- Start1880: tc-only consensus 0.878 beats all-consensus 0.865; best tc_rate gate picks tc-only (F1 0.878).
- Start1900: tc-only consensus 0.847 beats all-consensus 0.814; best tc_rate gate picks tc-only (F1 0.847).
- Start1920: tc-only consensus 0.772 below all-consensus 0.829; best tc_rate gate yields F1 0.830.
- Start1940: tc-only consensus 0.839 beats all-consensus 0.837; best tc_rate gate at 0.103 yields F1 0.847; tc_count >= 13 yields F1 0.847.
- Start1960: tc-only consensus 0.861 beats all-consensus 0.836; best tc_rate gate at 0.088 yields F1 0.861; tc_count >= 10 yields F1 0.861.
- Start1980: tc-only consensus 0.809 beats all-consensus 0.790; best tc_rate gate at 0.102 yields F1 0.809; tc_count >= 9 yields F1 0.809.
- Start2000: tc-only consensus 0.785 below all-consensus 0.829; best tc_rate gate at 0.317 yields F1 0.829; tc_count >= 31 yields F1 0.829.
- Start2020: tc-only consensus 0.877 below all-consensus 0.886; best tc_rate gate at 0.338 yields F1 0.886; tc_count gate picks all-consensus (F1 0.886).
- Start2040: tc-only consensus 0.812 beats all-consensus 0.811; best tc_rate gate picks tc-only (F1 0.812); tc_count gate picks tc-only (F1 0.812).
- Start2060: tc-only consensus 0.853 below all-consensus 0.864; best tc_rate gate at 0.329 yields F1 0.870; tc_count >= 23 yields F1 0.870.
- Start2080: tc-only consensus 0.866 beats all-consensus 0.840; best tc_rate gate picks tc-only (F1 0.866); tc_count gate picks tc-only (F1 0.866).
- Start2100: tc-only consensus 0.826 beats all-consensus 0.813; best tc_rate gate at 0.28 yields F1 0.826; tc_count gate picks tc-only (F1 0.826).
- Start2120: tc-only consensus 0.815 below all-consensus 0.818; best tc_rate gate at 0.539 yields F1 0.820; tc_count >= 30 yields F1 0.825.
- Start2140: tc-only consensus 0.799 below all-consensus 0.803; best tc_rate gate picks all-consensus (F1 0.803); tc_count gate picks all-consensus (F1 0.803).
- Start2160: tc-only consensus 0.824 beats all-consensus 0.819; best tc_rate gate at 0.516 yields F1 0.824; tc_count >= 26 yields F1 0.824.
- Start2180: tc-only consensus 0.860 beats all-consensus 0.857; best tc_rate gate at 0.371 yields F1 0.861; tc_count >= 43 yields F1 0.861.
- Start2200: tc-only consensus 0.789 below all-consensus 0.841; best tc_rate gate at 0.927 yields F1 0.841; tc_count >= 94 yields F1 0.841.
- Start2220: tc-only consensus 0.780 below all-consensus 0.787; best tc_rate gate at 0.087 yields F1 0.792; tc_count >= 11 yields F1 0.792.
- Start2240: tc-only consensus 0.882 beats all-consensus 0.878; best tc_rate gate at 0.327 yields F1 0.882; tc_count >= 18 yields F1 0.882.
- Start2260: tc-only consensus 0.790 below all-consensus 0.793; best tc_rate gate at 0.979 yields F1 0.793; tc_count >= 67 yields F1 0.793.
- Cross-split gate: tc_rate >= 0.333 (picked on 64-candidate merged set) improves 128-candidate combined F1 to 0.836.
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2140 yields F1 0.832 (below tc-only 0.833).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2160 yields F1 0.831 (below tc-only 0.832).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2180 yields F1 0.834 (below tc-only 0.834).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2200 yields F1 0.834 (below tc-only 0.834).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2220 yields F1 0.837 (above all-consensus 0.835).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2240 yields F1 0.838 (above all-consensus 0.837).
- Cross-split gate (tc_rate >= 0.333) on merged 1580–2260 yields F1 0.837 (above all-consensus 0.835).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2100, best gate uses pairwise <= 0.787 to choose tc-only, reaching F1 0.840 (slightly above tc-only 0.839).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2120, best gate uses pairwise <= 0.758 to choose tc-only, reaching F1 0.838 (above tc-only 0.836).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2140, best gate uses pairwise <= 0.762 to choose tc-only, reaching F1 0.834 (above tc-only 0.833).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2160, best gate uses pairwise <= 0.759 to choose tc-only, reaching F1 0.833 (above tc-only 0.832).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2180, best gate uses pairwise <= 0.764 to choose tc-only, reaching F1 0.835 (above tc-only 0.834).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2200, best gate uses pairwise <= 0.762 to choose tc-only, reaching F1 0.835 (above tc-only 0.834).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2220, best gate uses pairwise <= 0.770 to choose tc-only, reaching F1 0.839 (above all-consensus 0.835).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2240, best gate uses pairwise <= 0.772 to choose tc-only, reaching F1 0.840 (above all-consensus 0.837).
- Feature-extended tc gate (pairwise/pairwise_tc/consensus_margin): on merged 1580–2260, best gate uses pairwise <= 0.771 to choose tc-only, reaching F1 0.838 (above all-consensus 0.835).
- Fixed pairwise gate (threshold 0.787 from merged tuning) across slices 1580–2100: average F1 0.839 vs all-consensus 0.839 and tc-only 0.839; beats all-consensus in 14/27 slices and tc-only in 12/27 (choose tc-only 47.8% of records).
- Fixed pairwise gate (threshold 0.758 from merged tuning) across slices 1580–2120: average F1 0.842 vs all-consensus 0.838 and tc-only 0.838; beats all-consensus in 15/28 slices and tc-only in 15/28 (choose tc-only 36.1% of records).
- Fixed pairwise gate (threshold 0.762 from merged tuning) across slices 1580–2140: average F1 0.840 vs all-consensus 0.837 and tc-only 0.836; beats all-consensus in 15/29 slices and tc-only in 15/29 (choose tc-only 37.2% of records).
- Fixed pairwise gate (threshold 0.759 from merged tuning) across slices 1580–2160: average F1 0.840 vs all-consensus 0.836 and tc-only 0.836; beats all-consensus in 16/30 slices and tc-only in 16/30 (choose tc-only 36.6% of records).
- Fixed pairwise gate (threshold 0.764 from merged tuning) across slices 1580–2180: average F1 0.840 vs all-consensus 0.837 and tc-only 0.837; beats all-consensus in 16/31 slices and tc-only in 16/31 (choose tc-only 37.7% of records).
- Fixed pairwise gate (threshold 0.762 from merged tuning) across slices 1580–2200: average F1 0.840 vs all-consensus 0.837 and tc-only 0.837; beats all-consensus in 16/32 slices and tc-only in 17/32 (choose tc-only 37.5% of records).
- Fixed pairwise gate (threshold 0.770 from merged tuning) across slices 1580–2220: average F1 0.839 vs all-consensus 0.835 and tc-only 0.835; beats all-consensus in 16/33 slices and tc-only in 16/33 (choose tc-only 40.2% of records).
- Fixed pairwise gate (threshold 0.772 from merged tuning) across slices 1580–2240: average F1 0.840 vs all-consensus 0.837 and tc-only 0.836; beats all-consensus in 16/34 slices and tc-only in 16/34 (choose tc-only 40.2% of records).
- Fixed pairwise gate (threshold 0.771 from merged tuning) across slices 1580–2260: average F1 0.838 vs all-consensus 0.835 and tc-only 0.835; beats all-consensus in 16/35 slices and tc-only in 17/35 (choose tc-only 40.1% of records).
- Oracle gate (choose the better of all-consensus vs tc-only per record) reaches F1 0.851 on merged 1580–2100; tc-only is better on 22.1% of records, all-consensus on 16.0%, ties 61.9%.
- Learned tc gate (linear regression + 5-fold CV): F1 0.838 (base features), 0.839 with extra cycle/lenpen stats; still below tc-only 0.839 and pairwise gate 0.840.
- Learned tc gate on merged 1580–2260 (5-fold CV): F1 0.842 (base features), 0.841 with extra stats; tuned threshold peaks at 0.842 (avg thr -0.0015), beating pairwise gate 0.838.
- GBRT tc gate on merged 1580–2260: base F1 0.839 (tuned 0.839), extra stats F1 0.841 (tuned 0.841); still below ridge gate 0.842.
- GBRT sweep on merged 1580–2260 (base, tuned thresholds; n=100/200, lr=0.03/0.05, depth 2/3): best F1 0.840 at n=100 lr=0.03 depth=3, still below ridge 0.842.
- New script: `analysis_consensus_tc_gate_learned_slice_cv.py` (leave-one-slice-out learned gate).
- LOSO learned gate on 1580–2260: F1 0.842 (base features), 0.842 with tuned threshold (avg thr -0.0044), 0.841 with extra stats; beats all-consensus in 25/35 slices (base), tc-only in 22/35.
- LOSO GBRT gate on 1580–2260 (tuned thresholds): base F1 0.840; extra stats F1 0.842 (slightly below ridge tuned 0.842).
- LOSO GBRT gate with sweepbest params (n=100 lr=0.03 depth=3, extra stats) reaches F1 0.841; no improvement vs default GBRT.
- 2D grid gate (pairwise + tc_rate): best full-fit rule (pairwise<=0.787 or tc_rate>=0.626) reaches F1 0.841, but 5-fold CV drops to 0.837 (below tc-only).

### TC-gated consensus (64-candidate merged)

- Temp‑0.8 merged (start-index 740–1460, 172 records):
  - all-consensus F1 0.822; tc-only consensus 0.822.
  - gating by tc_rate >= 0.333 yields F1 0.827 (best), improving over all-consensus.
  - gating by tc_count >= 8 yields F1 0.825.
- Learned tc gate (linear regression + 5-fold CV) on merged 740–1460:
  - base features F1 0.821 (below all-consensus 0.822); extra stats F1 0.825 (above consensus), tuned extra stats F1 0.824 (avg thr -0.002).
- GBRT tc gate on merged 740–1460 (base features): F1 0.827 (tuned threshold avg thr -0.003), slightly above tc_rate gate 0.827 and ridge+extra 0.825.
- GBRT hyperparam sweep on merged 740–1460 (base, tuned threshold): best n=200 lr=0.05 depth=2 reaches F1 0.828 (avg thr 0.0036), a small edge over depth=3.
- GBRT sweep on merged 740–1460 with extra stats (tuned threshold): best n=100 lr=0.03 depth=3 reaches F1 0.828 (avg thr -0.0008), the strongest 64‑cand gate so far.
- LOSO learned gate across temp‑0.8 64‑candidate slices (start-index 740–1460, 701 records, 37 slices):
  - base features F1 0.824; extra stats F1 0.825; beats all-consensus in 27/37 slices and tc-only in 25/37.
- LOSO GBRT gate across temp‑0.8 64‑candidate slices (tuned thresholds):
  - base features F1 0.823; extra stats F1 0.824; beats all-consensus in 25–27/37 slices.
- LOSO GBRT gate with sweepbest params (n=100 lr=0.03 depth=3, extra stats) reaches F1 0.824; no gain over default GBRT.
- Pairwise gate on merged 740–1460 (threshold 0.942) collapses to tc-only (F1 0.822); fixed gate across slices chooses tc-only 99.9% of records (F1 0.822).
- New script: `analysis_consensus_tc_gate_transfer.py` (train gate on one split, evaluate on another).
- Transfer gate (train 128‑cand 1580–2260 → test 64‑cand 740–1460, tuned threshold): base F1 0.828; extra stats F1 0.828 (slightly lower, 0.8276); both beat all‑consensus 0.822.
- Transfer gate (train 64‑cand 740–1460 → test 128‑cand 1580–2260, tuned threshold): base F1 0.835 (no gain), extra stats F1 0.839 (beats all‑consensus 0.835).

### Consensus signal analysis (high-temp slices)

- New script: `analysis_consensus_signal.py` (correlates consensus gains with candidate-set properties).
- Combined temp‑0.6 slices (16/32/64 candidates, 70 records):
  - baseline F1 0.765 → consensus 0.808 → oracle 0.895
  - consensus improvement correlates with candidate count (r=0.345) and low baseline F1 (r=-0.398)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.558)
- Updated with start-index 720 (16/32/64/64 candidates, 83 records):
  - baseline F1 0.760 → consensus 0.803 → oracle 0.893
  - consensus improvement correlates most with low baseline F1 (r=-0.399) and candidate count (r=0.198)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.531)
- 64-candidate slice only (19 records): consensus improvement rate 0.684; correlations strongest with low baseline F1 (r=-0.503) and higher consensus margin (r=0.416).
- 64-candidate slices combined (start-index 700 + 720, 36 records):
  - baseline F1 0.749 → consensus 0.804 → oracle 0.901; improve rate 0.556
  - consensus improvement correlates with low baseline F1 (r=-0.390) and consensus margin (r=0.215)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.530)
- Temp‑0.8 slice (start-index 740, 18 records):
  - baseline F1 0.746 → consensus 0.797 → oracle 0.936; improve rate 0.722
  - consensus improvement correlates with low baseline F1 (r=-0.534) and lower tc rate (r=-0.366)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.634)
- Temp‑0.8 slice (start-index 760, 17 records):
  - baseline F1 0.702 → consensus 0.794 → oracle 0.924; improve rate 0.706
  - consensus improvement correlates with low baseline F1 (r=-0.854) and candidate count (r=0.209)
- Temp‑0.8 slices combined (start-index 740 + 760, 32 records):
  - baseline F1 0.720 → consensus 0.788 → oracle 0.930; improve rate 0.688
  - consensus improvement correlates with low baseline F1 (r=-0.776)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.302)
- Temp‑0.8 slice (start-index 780, 20 records):
  - baseline F1 0.759 → consensus 0.826 → oracle 0.926; improve rate 0.800
  - consensus improvement correlates with low baseline F1 (r=-0.447) and higher pairwise agreement (r=0.378)
- Temp‑0.8 slices combined (start-index 740 + 760 + 780, 45 records):
  - baseline F1 0.730 → consensus 0.803 → oracle 0.928; improve rate 0.733
  - consensus improvement correlates with low baseline F1 (r=-0.744)
- Temp‑0.8 slice (start-index 800, 18 records):
  - baseline F1 0.772 → consensus 0.793 → oracle 0.914; improve rate 0.611
  - consensus improvement correlates with low baseline F1 (r=-0.599) and higher pairwise agreement (r=0.356)
- Temp‑0.8 slices combined (start-index 740 + 760 + 780 + 800, 57 records):
  - baseline F1 0.738 → consensus 0.802 → oracle 0.924; improve rate 0.719
  - consensus improvement correlates with low baseline F1 (r=-0.715)
- Temp‑0.8 slice (start-index 820, 19 records):
  - baseline F1 0.815 → consensus 0.835 → oracle 0.925; improve rate 0.526
  - consensus improvement correlates with higher pairwise agreement (r=0.451) and tc agreement (r=0.728)
- Temp‑0.8 slice (start-index 840, 19 records):
  - baseline F1 0.822 → consensus 0.824 → oracle 0.946; improve rate 0.368
  - consensus improvement correlates with consensus margin (r=0.574)
- Temp‑0.8 slice (start-index 860, 20 records):
  - baseline F1 0.817 → consensus 0.841 → oracle 0.919; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.477)
- Temp‑0.8 slice (start-index 880, 20 records):
  - baseline F1 0.733 → consensus 0.794 → oracle 0.918; improve rate 0.600
  - consensus improvement correlates with low baseline F1 (r=-0.712)
- Temp‑0.8 slice (start-index 900, 19 records):
  - baseline F1 0.702 → consensus 0.788 → oracle 0.921; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.819)
- Temp‑0.8 slice (start-index 920, 20 records):
  - baseline F1 0.785 → consensus 0.870 → oracle 0.928; improve rate 0.750
  - consensus improvement correlates with low baseline F1 (r=-0.772) and higher margin (r=0.569)
- Temp‑0.8 slice (start-index 940, 18 records):
  - baseline F1 0.779 → consensus 0.841 → oracle 0.913; improve rate 0.611
  - consensus improvement correlates with low baseline F1 (r=-0.746) and higher margin (r=0.446)
- Temp‑0.8 slice (start-index 960, 17 records):
  - baseline F1 0.804 → consensus 0.837 → oracle 0.936; improve rate 0.647
  - consensus improvement correlates with low baseline F1 (r=-0.589)
- Temp‑0.8 slice (start-index 980, 20 records):
  - baseline F1 0.753 → consensus 0.810 → oracle 0.928; improve rate 0.650
  - consensus improvement correlates with low baseline F1 (r=-0.540)
- Temp‑0.8 slice (start-index 1000, 19 records):
  - baseline F1 0.831 → consensus 0.861 → oracle 0.955; improve rate 0.526
  - consensus improvement correlates with low baseline F1 (r=-0.573)
- Temp‑0.8 slice (start-index 1020, 17 records):
  - baseline F1 0.790 → consensus 0.846 → oracle 0.959; improve rate 0.529
  - consensus improvement correlates with low baseline F1 (r=-0.656) and lower pairwise agreement (r=-0.460)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.703)
- Temp‑0.8 slice (start-index 1040, 20 records):
  - baseline F1 0.760 → consensus 0.768 → oracle 0.924; improve rate 0.600
  - consensus improvement correlates with low baseline F1 (r=-0.477) and higher candidate count (r=0.388)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.507)
- Temp‑0.8 slice (start-index 1060, 18 records):
  - baseline F1 0.788 → consensus 0.820 → oracle 0.924; improve rate 0.500
  - consensus improvement correlates with low baseline F1 (r=-0.536) and lower pairwise agreement (r=-0.272)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.455)
- Temp‑0.8 slice (start-index 1080, 19 records):
  - baseline F1 0.784 → consensus 0.818 → oracle 0.933; improve rate 0.526
  - consensus improvement correlates with low baseline F1 (r=-0.322) and lower pairwise agreement (r=-0.198)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.702)
- Temp‑0.8 slice (start-index 1100, 20 records):
  - baseline F1 0.766 → consensus 0.801 → oracle 0.903; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.225) and lower pairwise agreement (r=-0.287)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.576)
- Temp‑0.8 slice (start-index 1120, 18 records):
  - baseline F1 0.803 → consensus 0.862 → oracle 0.933; improve rate 0.667
  - consensus improvement correlates with low baseline F1 (r=-0.536) and higher margin (r=0.262)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.684)
- Temp‑0.8 slice (start-index 1140, 20 records):
  - baseline F1 0.748 → consensus 0.767 → oracle 0.906; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.449) and higher candidate count (r=0.219)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.338)
- Temp‑0.8 slice (start-index 1160, 19 records):
  - baseline F1 0.792 → consensus 0.816 → oracle 0.941; improve rate 0.421
  - consensus improvement correlates with low baseline F1 (r=-0.316) and lower tc rate (r=-0.381)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.632)
- Temp‑0.8 slice (start-index 1180, 19 records):
  - baseline F1 0.737 → consensus 0.803 → oracle 0.917; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.705) and lower pairwise agreement (r=-0.435)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.671)
- Temp‑0.8 slice (start-index 1200, 20 records):
  - baseline F1 0.824 → consensus 0.855 → oracle 0.938; improve rate 0.600
  - consensus improvement correlates with low baseline F1 (r=-0.424) and higher pairwise agreement (r=0.099)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.278)
- Temp‑0.8 slice (start-index 1220, 19 records):
  - baseline F1 0.752 → consensus 0.802 → oracle 0.900; improve rate 0.526
  - consensus improvement correlates with low baseline F1 (r=-0.532) and higher margin (r=0.706)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.370)
- Temp‑0.8 slice (start-index 1240, 20 records):
  - baseline F1 0.758 → consensus 0.808 → oracle 0.928; improve rate 0.450
  - consensus improvement correlates with low baseline F1 (r=-0.487)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.294)
- Temp‑0.8 slice (start-index 1260, 19 records):
  - baseline F1 0.728 → consensus 0.804 → oracle 0.915; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.405)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.400)
- Temp‑0.8 slice (start-index 1280, 19 records):
  - baseline F1 0.746 → consensus 0.817 → oracle 0.925; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.929)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.404)
- Temp‑0.8 slice (start-index 1300, 19 records):
  - baseline F1 0.792 → consensus 0.817 → oracle 0.917; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.275)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.225)
- Temp‑0.8 slice (start-index 1320, 20 records):
  - baseline F1 0.773 → consensus 0.836 → oracle 0.931; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.888)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.192)
- Temp‑0.8 slice (start-index 1340, 18 records):
  - baseline F1 0.835 → consensus 0.856 → oracle 0.936; improve rate 0.500
  - consensus improvement correlates with low baseline F1 (r=-0.676) and higher pairwise agreement (r=0.242)
  - oracle improvement correlates weakly with diversity (avg pairwise F1 r=-0.034)
- Temp‑0.8 slice (start-index 1360, 20 records):
  - baseline F1 0.731 → consensus 0.784 → oracle 0.912; improve rate 0.850
  - consensus improvement correlates with low baseline F1 (r=-0.317)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.559)
- Temp‑0.8 slice (start-index 1380, 19 records):
  - baseline F1 0.768 → consensus 0.832 → oracle 0.927; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.651)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.586)
- Temp‑0.8 slice (start-index 1400, 17 records):
  - baseline F1 0.832 → consensus 0.846 → oracle 0.938; improve rate 0.471
  - consensus improvement correlates with low baseline F1 (r=-0.610) and higher pairwise agreement (r=0.712)
  - oracle improvement correlates with higher pairwise agreement (r=0.273)
- Temp‑0.8 slice (start-index 1420, 19 records):
  - baseline F1 0.816 → consensus 0.875 → oracle 0.942; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.563) and higher margin (r=0.192)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.312)
- Temp‑0.8 slice (start-index 1440, 18 records):
  - baseline F1 0.788 → consensus 0.828 → oracle 0.917; improve rate 0.611
  - consensus improvement correlates with low baseline F1 (r=-0.413) and higher margin (r=0.198)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.549)
- Temp‑0.8 slice (start-index 1460, 20 records):
  - baseline F1 0.755 → consensus 0.820 → oracle 0.909; improve rate 0.700
  - consensus improvement correlates with low baseline F1 (r=-0.643) and higher margin (r=0.356)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.419)
- Temp‑0.8 slice (start-index 1480, 19 records):
  - baseline F1 0.775 → consensus 0.808 → oracle 0.920; improve rate 0.632
  - consensus improvement correlates with higher margin (r=0.704) and lower tc rate (r=-0.473)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.658)
- Temp‑0.8 slice (start-index 1500, 20 records):
  - baseline F1 0.788 → consensus 0.825 → oracle 0.931; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.533) and lower tc rate (r=-0.451)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.432)
- Temp‑0.8 slice (start-index 1520, 17 records):
  - baseline F1 0.779 → consensus 0.863 → oracle 0.939; improve rate 0.588
  - consensus improvement correlates with low baseline F1 (r=-0.913) and lower pairwise agreement (r=-0.376)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.473)
- Temp‑0.8 slice (start-index 1540, 18 records):
  - baseline F1 0.789 → consensus 0.833 → oracle 0.946; improve rate 0.722
  - consensus improvement correlates with higher pairwise agreement (r=0.412)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.422)
- Temp‑0.8 slice (start-index 1560, 17 records):
  - baseline F1 0.796 → consensus 0.832 → oracle 0.939; improve rate 0.706
  - consensus improvement correlates with low baseline F1 (r=-0.541) and lower tc rate (r=-0.388)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.659)
- Temp‑0.8 slice (start-index 1580, 20 records, 128 candidates):
  - baseline F1 0.808 → consensus 0.875 → oracle 0.947; improve rate 0.700
  - consensus improvement correlates with low baseline F1 (r=-0.818) and higher margin (r=0.141)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.281)
- Temp‑0.8 slice (start-index 1600, 19 records, 128 candidates):
  - baseline F1 0.788 → consensus 0.872 → oracle 0.957; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.878) and lower tc rate (r=-0.469)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.317)
- Temp‑0.8 slice (start-index 1620, 20 records, 128 candidates):
  - baseline F1 0.677 → consensus 0.774 → oracle 0.912; improve rate 0.500
  - consensus improvement correlates with low baseline F1 (r=-0.779) and lower pairwise agreement (r=-0.497)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.719)
- Temp‑0.8 slice (start-index 1640, 20 records, 128 candidates):
  - baseline F1 0.796 → consensus 0.861 → oracle 0.929; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.751)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.165)
- Temp‑0.8 slice (start-index 1660, 19 records, 128 candidates):
  - baseline F1 0.730 → consensus 0.839 → oracle 0.935; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.832) and low pairwise agreement (r=-0.632)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.793)
- Temp‑0.8 slice (start-index 1680, 17 records, 128 candidates):
  - baseline F1 0.802 → consensus 0.836 → oracle 0.944; improve rate 0.647
  - consensus improvement correlates with low baseline F1 (r=-0.352) and higher pairwise agreement (r=0.233)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.495)
- Temp‑0.8 slice (start-index 1700, 19 records, 128 candidates):
  - baseline F1 0.767 → consensus 0.799 → oracle 0.944; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.679) and higher margin (r=0.197)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.617)
- Temp‑0.8 slice (start-index 1720, 18 records, 128 candidates):
  - baseline F1 0.770 → consensus 0.852 → oracle 0.925; improve rate 0.667
  - consensus improvement correlates with low baseline F1 (r=-0.714) and low pairwise agreement (r=-0.596)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.512)
- Temp‑0.8 slice (start-index 1740, 18 records, 128 candidates):
  - baseline F1 0.810 → consensus 0.825 → oracle 0.939; improve rate 0.556
  - consensus improvement correlates with higher pairwise agreement (r=0.551)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.210)
- Temp‑0.8 slice (start-index 1760, 18 records, 128 candidates):
  - baseline F1 0.781 → consensus 0.843 → oracle 0.937; improve rate 0.722
  - consensus improvement correlates with low baseline F1 (r=-0.686) and lower pairwise agreement (r=-0.450)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.601)
- Temp‑0.8 slice (start-index 1780, 20 records, 128 candidates):
  - baseline F1 0.823 → consensus 0.872 → oracle 0.953; improve rate 0.650
  - consensus improvement correlates with low baseline F1 (r=-0.434) and higher pairwise agreement among tc candidates (r=0.351)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.112)
- Temp‑0.8 slice (start-index 1800, 19 records, 128 candidates):
  - baseline F1 0.822 → consensus 0.860 → oracle 0.950; improve rate 0.526
  - consensus improvement correlates with low baseline F1 (r=-0.381) and higher tc rate (r=0.231)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.190)
- Temp‑0.8 slice (start-index 1820, 18 records, 128 candidates):
  - baseline F1 0.777 → consensus 0.843 → oracle 0.940; improve rate 0.611
  - consensus improvement correlates with low baseline F1 (r=-0.604) and lower tc rate (r=-0.793)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.516)
- Temp‑0.8 slice (start-index 1840, 18 records, 128 candidates):
  - baseline F1 0.759 → consensus 0.853 → oracle 0.952; improve rate 0.778
  - consensus improvement correlates with low baseline F1 (r=-0.653) and lower pairwise agreement (r=-0.250)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.732)
- Temp‑0.8 slice (start-index 1860, 18 records, 128 candidates):
  - baseline F1 0.786 → consensus 0.823 → oracle 0.933; improve rate 0.667
  - consensus improvement correlates with low baseline F1 (r=-0.319) and lower tc rate (r=-0.316)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.566)
- Temp‑0.8 slice (start-index 1880, 20 records, 128 candidates):
  - baseline F1 0.818 → consensus 0.865 → oracle 0.951; improve rate 0.450
  - consensus improvement correlates with low baseline F1 (r=-0.762) and higher tc rate (r=0.286)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.657)
- Temp‑0.8 slice (start-index 1900, 20 records, 128 candidates):
  - baseline F1 0.785 → consensus 0.814 → oracle 0.942; improve rate 0.600
  - consensus improvement correlates with low baseline F1 (r=-0.602) and higher tc rate (r=0.050)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.475)
- Temp‑0.8 slice (start-index 1920, 18 records, 128 candidates):
  - baseline F1 0.771 → consensus 0.829 → oracle 0.939; improve rate 0.667
  - consensus improvement correlates with low baseline F1 (r=-0.400) and higher candidate count (r=0.452)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.440)
- Temp‑0.8 slice (start-index 1940, 20 records, 128 candidates):
  - baseline F1 0.785 → consensus 0.837 → oracle 0.938; improve rate 0.400
  - consensus improvement correlates with low baseline F1 (r=-0.772) and higher tc rate (r=0.272)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.171)
- Temp‑0.8 slice (start-index 1960, 19 records, 128 candidates):
  - baseline F1 0.764 → consensus 0.836 → oracle 0.946; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.565) and higher tc rate (r=0.539)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.492)
- Temp‑0.8 slice (start-index 1980, 20 records, 128 candidates):
  - baseline F1 0.749 → consensus 0.790 → oracle 0.928; improve rate 0.550
  - consensus improvement correlates with lower tc rate (r=-0.644) and low baseline F1 (r=-0.152)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.411)
- Temp‑0.8 slice (start-index 2000, 19 records, 128 candidates):
  - baseline F1 0.761 → consensus 0.829 → oracle 0.919; improve rate 0.684
  - consensus improvement correlates with low baseline F1 (r=-0.699) and higher tc rate (r=0.260)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.460)
- Temp‑0.8 slice (start-index 2020, 19 records, 128 candidates):
  - baseline F1 0.843 → consensus 0.886 → oracle 0.952; improve rate 0.684
  - consensus improvement correlates with low baseline F1 (r=-0.555) and higher pairwise agreement (r=0.344)
  - oracle improvement correlates weakly with diversity (avg pairwise F1 r=-0.047)
- Temp‑0.8 slice (start-index 2040, 20 records, 128 candidates):
  - baseline F1 0.714 → consensus 0.811 → oracle 0.929; improve rate 0.700
  - consensus improvement correlates with low baseline F1 (r=-0.609) and lower tc rate (r=-0.460)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.743)
- Temp‑0.8 slice (start-index 2060, 19 records, 128 candidates):
  - baseline F1 0.755 → consensus 0.864 → oracle 0.944; improve rate 0.684
  - consensus improvement correlates with low baseline F1 (r=-0.823) and lower tc rate (r=-0.467)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.645)
- Temp‑0.8 slice (start-index 2080, 19 records, 128 candidates):
  - baseline F1 0.779 → consensus 0.840 → oracle 0.945; improve rate 0.789
  - consensus improvement correlates with low baseline F1 (r=-0.738) and weakly with tc rate (r=0.020)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.602)
- Temp‑0.8 slice (start-index 2100, 19 records, 128 candidates):
  - baseline F1 0.801 → consensus 0.813 → oracle 0.933; improve rate 0.368
  - consensus improvement correlates with low baseline F1 (r=-0.715) and candidate count (r=0.626)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.677)
- Temp‑0.8 slice (start-index 2120, 19 records, 128 candidates):
  - baseline F1 0.783 → consensus 0.816 → oracle 0.924; improve rate 0.556
  - consensus improvement correlates with lower consensus margin (r=-0.482) and lower tc rate (r=-0.387)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.318)
- Temp‑0.8 slice (start-index 2140, 19 records, 128 candidates):
  - baseline F1 0.748 → consensus 0.803 → oracle 0.907; improve rate 0.474
  - consensus improvement correlates with low baseline F1 (r=-0.525) and higher tc rate (r=0.309)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.447)
- Temp‑0.8 slice (start-index 2160, 20 records, 128 candidates):
  - baseline F1 0.754 → consensus 0.819 → oracle 0.918; improve rate 0.550
  - consensus improvement correlates with low baseline F1 (r=-0.633) and lower pairwise agreement (r=-0.396)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.366)
- Temp‑0.8 slice (start-index 2180, 20 records, 128 candidates):
  - baseline F1 0.758 → consensus 0.857 → oracle 0.939; improve rate 0.700
  - consensus improvement correlates with low baseline F1 (r=-0.914) and higher candidate count (r=0.318)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.274)
- Temp‑0.8 slice (start-index 2200, 20 records, 128 candidates):
  - baseline F1 0.784 → consensus 0.841 → oracle 0.927; improve rate 0.700
  - consensus improvement correlates with low baseline F1 (r=-0.497) and lower pairwise_tc (r=-0.616)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.393)
- Temp‑0.8 slice (start-index 2220, 19 records, 128 candidates):
  - baseline F1 0.761 → consensus 0.787 → oracle 0.910; improve rate 0.579
  - consensus improvement correlates with low baseline F1 (r=-0.476) and higher consensus margin (r=0.262)
  - oracle improvement correlates weakly with candidate count (r=0.249)
- Temp‑0.8 slice (start-index 2240, 20 records, 128 candidates):
  - baseline F1 0.835 → consensus 0.878 → oracle 0.932; improve rate 0.450
  - consensus improvement correlates with low baseline F1 (r=-0.640) and higher consensus margin (r=0.415)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.493)
- Temp‑0.8 slice (start-index 2260, 18 records, 128 candidates):
  - baseline F1 0.752 → consensus 0.793 → oracle 0.895; improve rate 0.778
  - consensus improvement correlates with lower tc rate (r=-0.416) and higher candidate count (r=0.299)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.271)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640, 64 records):
  - baseline F1 0.765 → consensus 0.842 → oracle 0.933; improve rate 0.594
  - consensus improvement correlates with low baseline F1 (r=-0.749)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660, 98 records):
  - baseline F1 0.760 → consensus 0.844 → oracle 0.936; improve rate 0.592
  - consensus improvement correlates with low baseline F1 (r=-0.786)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680, 115 records):
  - baseline F1 0.766 → consensus 0.843 → oracle 0.937; improve rate 0.600
  - consensus improvement correlates with low baseline F1 (r=-0.768)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700, 134 records):
  - baseline F1 0.766 → consensus 0.837 → oracle 0.938; improve rate 0.604
  - consensus improvement correlates with low baseline F1 (r=-0.749)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720, 152 records):
  - baseline F1 0.767 → consensus 0.838 → oracle 0.937; improve rate 0.612
  - consensus improvement correlates with low baseline F1 (r=-0.745)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740, 170 records):
  - baseline F1 0.771 → consensus 0.837 → oracle 0.937; improve rate 0.606
  - consensus improvement correlates with low baseline F1 (r=-0.724)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780, 113 records):
  - baseline F1 0.772 → consensus 0.839 → oracle 0.936; improve rate 0.628
  - consensus improvement correlates with low baseline F1 (r=-0.715)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800, 116 records):
  - baseline F1 0.772 → consensus 0.839 → oracle 0.937; improve rate 0.629
  - consensus improvement correlates with low baseline F1 (r=-0.710)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820, 121 records):
  - baseline F1 0.770 → consensus 0.836 → oracle 0.935; improve rate 0.620
  - consensus improvement correlates with low baseline F1 (r=-0.688)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840, 126 records):
  - baseline F1 0.772 → consensus 0.838 → oracle 0.936; improve rate 0.627
  - consensus improvement correlates with low baseline F1 (r=-0.684)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860, 129 records):
  - baseline F1 0.772 → consensus 0.838 → oracle 0.935; improve rate 0.628
  - consensus improvement correlates with low baseline F1 (r=-0.683)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880, 134 records):
  - baseline F1 0.769 → consensus 0.837 → oracle 0.935; improve rate 0.627
  - consensus improvement correlates with low baseline F1 (r=-0.689)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900, 142 records):
  - baseline F1 0.772 → consensus 0.837 → oracle 0.936; improve rate 0.627
  - consensus improvement correlates with low baseline F1 (r=-0.684)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920, 144 records):
  - baseline F1 0.771 → consensus 0.839 → oracle 0.937; improve rate 0.632
  - consensus improvement correlates with low baseline F1 (r=-0.686)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940, 147 records):
  - baseline F1 0.772 → consensus 0.838 → oracle 0.936; improve rate 0.626
  - consensus improvement correlates with low baseline F1 (r=-0.683)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960, 150 records):
  - baseline F1 0.771 → consensus 0.837 → oracle 0.935; improve rate 0.627
  - consensus improvement correlates with low baseline F1 (r=-0.673)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980, 151 records):
  - baseline F1 0.769 → consensus 0.834 → oracle 0.936; improve rate 0.623
  - consensus improvement correlates with low baseline F1 (r=-0.649)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000, 152 records):
  - baseline F1 0.768 → consensus 0.833 → oracle 0.934; improve rate 0.625
  - consensus improvement correlates with low baseline F1 (r=-0.648)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020, 156 records):
  - baseline F1 0.771 → consensus 0.835 → oracle 0.935; improve rate 0.628
  - consensus improvement correlates with low baseline F1 (r=-0.647)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040, 158 records):
  - baseline F1 0.768 → consensus 0.836 → oracle 0.935; improve rate 0.633
  - consensus improvement correlates with low baseline F1 (r=-0.654)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060, 159 records):
  - baseline F1 0.768 → consensus 0.837 → oracle 0.935; improve rate 0.635
  - consensus improvement correlates with low baseline F1 (r=-0.653)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080, 161 records):
  - baseline F1 0.768 → consensus 0.836 → oracle 0.935; improve rate 0.640
  - consensus improvement correlates with low baseline F1 (r=-0.650)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100, 163 records):
  - baseline F1 0.766 → consensus 0.834 → oracle 0.935; improve rate 0.638
  - consensus improvement correlates with low baseline F1 (r=-0.645)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120, 164 records):
  - baseline F1 0.767 → consensus 0.835 → oracle 0.935; improve rate 0.640
  - consensus improvement correlates with low baseline F1 (r=-0.644)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140, 201 records):
  - baseline F1 0.767 → consensus 0.830 → oracle 0.931; improve rate 0.612
  - consensus improvement correlates with low baseline F1 (r=-0.606)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.515)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160, 221 records):
  - baseline F1 0.766 → consensus 0.829 → oracle 0.930; improve rate 0.606
  - consensus improvement correlates with low baseline F1 (r=-0.606)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.504)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180, 241 records):
  - baseline F1 0.765 → consensus 0.831 → oracle 0.931; improve rate 0.614
  - consensus improvement correlates with low baseline F1 (r=-0.661)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.464)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200, 261 records):
  - baseline F1 0.766 → consensus 0.832 → oracle 0.931; improve rate 0.621
  - consensus improvement correlates with low baseline F1 (r=-0.654)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.461)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220, 630 records):
  - baseline F1 0.776 → consensus 0.835 → oracle 0.936; improve rate 0.610
  - consensus improvement correlates with low baseline F1 (r=-0.643)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.452)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240, 650 records):
  - baseline F1 0.778 → consensus 0.837 → oracle 0.936; improve rate 0.605
  - consensus improvement correlates with low baseline F1 (r=-0.642)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.458)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760 + 1780 + 1800 + 1820 + 1840 + 1860 + 1880 + 1900 + 1920 + 1940 + 1960 + 1980 + 2000 + 2020 + 2040 + 2060 + 2080 + 2100 + 2120 + 2140 + 2160 + 2180 + 2200 + 2220 + 2240 + 2260, 668 records):
  - baseline F1 0.777 → consensus 0.835 → oracle 0.935; improve rate 0.609
  - consensus improvement correlates with low baseline F1 (r=-0.627)
  - oracle improvement correlates with diversity (avg pairwise F1 r=-0.453)
- Temp‑0.8 128-candidate slices combined (start-index 1580 + 1600 + 1620 + 1640 + 1660 + 1680 + 1700 + 1720 + 1740 + 1760, 108 records):
  - baseline F1 0.770 → consensus 0.837 → oracle 0.935; improve rate 0.611
  - consensus improvement correlates with low baseline F1 (r=-0.718)
- Temp‑0.8 slices combined (start-index 740 + 760 + 780 + 800 + 820 + 840 + 860 + 880 + 900 + 920 + 940 + 960 + 980 + 1000 + 1020 + 1040 + 1060 + 1080 + 1100 + 1120 + 1140 + 1160 + 1180 + 1200 + 1220 + 1240 + 1260 + 1280 + 1300 + 1320 + 1340 + 1360 + 1380 + 1400 + 1420 + 1440 + 1460, 172 records):
  - baseline F1 0.757 → consensus 0.822 → oracle 0.926; improve rate 0.669 (no new unique ids vs 1440)
  - consensus improvement correlates with low baseline F1 (r=-0.694)
  - note: adding start-index 1480 yields no new unique ids vs 1460 (count stays 172)
- Temp‑0.8 slices combined (start-index 740 + 760 + 780 + 800 + 820 + 840 + 860 + 880 + 900 + 920 + 940 + 960 + 980 + 1000 + 1020 + 1040 + 1060 + 1080 + 1100 + 1120 + 1140 + 1160 + 1180 + 1200 + 1220 + 1240 + 1260 + 1280 + 1300 + 1320 + 1340 + 1360 + 1380 + 1400 + 1420 + 1440 + 1460 + 1480 + 1500 + 1520 + 1540 + 1560, 173 records):
  - baseline F1 0.756 → consensus 0.821 → oracle 0.925; improve rate 0.671
  - consensus improvement correlates with low baseline F1 (r=-0.692)
  - note: adding start-index 1520 yields no new unique ids vs 1500 (count stays 173)
  - note: adding start-index 1540 yields no new unique ids vs 1520 (count stays 173)
  - note: adding start-index 1560 yields no new unique ids vs 1540 (count stays 173)

### Consensus gating sweep

- New script: `analysis_consensus_gate_sweep.py` (chooses baseline vs consensus based on a feature threshold).
- Combined temp‑0.6 slices: gating on pairwise agreement or tc_rate never beats always‑consensus; best F1 occurs at thresholds that effectively always choose consensus (F1 0.808, tc 0.843).
- Extended gating features: baseline_cycle and baseline_lenpen.
- Temp‑0.8 combined slices: gating by baseline cycle or length‑penalized score never beats always‑consensus; best F1 0.809 matches always‑consensus at high thresholds.
- Temp‑0.8 combined slices: gating by consensus margin (choose consensus when margin is high) also never beats always‑consensus; best F1 0.809.
- Temp‑0.8 merged 128-candidate slices (start-index 1580–2100): gating across count/tc_rate/pairwise/pairwise_tc/consensus_margin/baseline_cycle/baseline_len/baseline_lenpen never beats always‑consensus; best F1 0.834.

### Consensus all vs tc-only

- New script: `analysis_consensus_tc_compare.py`.
- 64-candidate slice: all-consensus F1 0.853 vs tc-only 0.838; tc-only wins only 10.5% of records.
- Combined temp‑0.6 slices: all-consensus F1 0.808 vs tc-only 0.807; tc-only wins 11.8% of records.

### Forward-score rerank (NL → Lean logprob)

- New script: `analysis_forward_rerank.py` (scores candidates with a forward LM and sweeps cycle-weight + length penalty).
- Temp‑0.8 slice (start-index 1460, 20 records, Qwen2.5‑1.5B forward):
  - baseline F1 0.755; forward-only best F1 0.749 (alpha 0.0, beta 0.0)
  - best forward+cycle: alpha 0.002, beta 0.5 → all F1 0.781, tc F1 0.799, tc rate 1.000
- Temp‑0.8 merged (start-index 740–1460, 172 records, Qwen2.5‑1.5B forward):
  - baseline F1 0.757; best forward+cycle all F1 0.754 (alpha 0.002, beta 1.0), below baseline
- Temp‑0.8 slice (start-index 1460, 20 records, Kimina‑7B forward):
  - baseline F1 0.755; forward-only best all F1 0.787 (alpha 0.0, beta 0.0)
  - forward+cycle is slightly worse than forward-only across this slice
- Temp‑0.8 merged (start-index 740–1460, 172 records, Kimina‑7B forward, CUDA_VISIBLE_DEVICES=0):
  - baseline F1 0.757; best forward-only all F1 0.806 (alpha 0.001, beta 0.0), tc F1 0.807, tc rate 0.994
  - still below consensus (F1 0.822) but a solid gain over baseline
- Temp‑0.8 merged (start-index 740–1460, 172 records, Qwen2.5‑7B forward, CUDA_VISIBLE_DEVICES=1):
  - baseline F1 0.757; best forward+cycle all F1 0.771 (alpha 0.002, beta 1.0), tc F1 0.783, tc rate 0.994
  - weaker than Kimina‑7B forward and below consensus
- Temp‑0.8 merged (start-index 740–1460, 172 records, Qwen2.5‑3B forward, CUDA_VISIBLE_DEVICES=2):
  - baseline F1 0.757; best forward+cycle all F1 0.758 (alpha 0.001, beta 1.0), tc F1 0.768, tc rate 0.994
  - only marginally above baseline, below Qwen‑7B and Kimina‑7B
- Temp‑0.6 merged (start-index 600–700, 70 records, Kimina‑7B forward, CUDA_VISIBLE_DEVICES=0):
  - baseline F1 0.765; best forward+cycle all F1 0.790 (alpha 0.002, beta 0.0), tc F1 0.793, tc rate 0.971
  - still below consensus (F1 0.808)

### Learned reranker (forward feature)

- Added optional `--forward-feature` to `analysis_learned_reranker.py`.
- Temp‑0.8 merged (start-index 740–1460, 172 records, Kimina‑7B forward scores):
  - baseline F1 0.757 → learned F1 0.825 (with baseline+forward features), tc F1 0.823, tc rate 0.959
  - baseline+cycle+consensus features without forward: learned F1 0.825 (forward adds ~+0.0003)
  - learned reranker now edges consensus on this split (F1 0.825 vs 0.822)
- Temp‑0.8 merged (start-index 740–1460, 172 records, Kimina‑7B forward, no baseline feature):
  - learned F1 0.826, tc F1 0.824, tc rate 0.965 (slightly better than with baseline feature)
- Temp‑0.8 merged (start-index 740–1460, 172 records, no forward + no baseline features):
  - learned F1 0.828, tc F1 0.826, tc rate 0.965 (best so far; forward feature not needed)
- Temp‑0.8 merged (start-index 740–1460, 172 records, Qwen2.5‑7B forward scores):
  - baseline F1 0.757 → learned F1 0.824 (with baseline+forward features), tc F1 0.822, tc rate 0.959
  - slightly below Kimina‑7B forward learned result
- Temp‑0.8 merged (start-index 740–1460, 172 records, Qwen2.5‑3B forward scores):
  - baseline F1 0.757 → learned F1 0.824 (with baseline+forward features), tc F1 0.823, tc rate 0.959
  - similar to Qwen‑7B learned, below Kimina‑7B learned
- Train/test split (index < 1100 train, >= 1100 test; Kimina‑7B forward):
  - test baseline F1 0.759; test consensus F1 0.821; learned F1 0.822 (forward adds ~+0.0005)
  - learned reranker slightly beats consensus on held-out set
- Train/test split (index < 1100 train, >= 1100 test; Kimina‑7B forward, no baseline feature):
  - learned F1 0.823, tc F1 0.824 (small gain over baseline-feature model)
- Train/test split (index < 1100 train, >= 1100 test; no forward + no baseline):
  - learned F1 0.823, tc F1 0.824 (same as forward; forward adds no benefit)
- Cross‑temperature transfer (Kimina‑7B forward, no baseline feature):
  - train temp‑0.6 (start 600–700) → test temp‑0.8 (start 740–1460): learned F1 0.825, tc F1 0.821 (beats consensus 0.822)
  - train temp‑0.8 (start 740–1460) → test temp‑0.6 (start 600–700): learned F1 0.812, tc F1 0.815 (beats consensus 0.808)
- Cross‑temperature transfer (no forward + no baseline features):
  - train temp‑0.6 → test temp‑0.8: learned F1 0.824, tc F1 0.820 (on par with forward transfer)
- Temp‑0.8 slice (start-index 1480, 19 records, no forward + no baseline):
  - CV learned F1 0.807, tc F1 0.807, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460) → test start 1480: learned F1 0.811, tc F1 0.811
  - transfer with no-cycle features (length+consensus+typecheck): learned F1 0.821
- Temp‑0.8 slice (start-index 1500, 20 records, no forward + no baseline):
  - CV learned (no-cycle) F1 0.818, tc F1 0.818, tc rate 0.950
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1500: learned F1 0.820
- Temp‑0.8 slice (start-index 1520, 17 records, no forward + no baseline):
  - CV learned (no-cycle) F1 0.862, tc F1 0.871, tc rate 0.588
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1520: learned F1 0.870
- Temp‑0.8 slice (start-index 1540, 18 records, no forward + no baseline):
  - CV learned (no-cycle) F1 0.849, tc F1 0.845, tc rate 0.833
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1540: learned F1 0.838
- Temp‑0.8 slice (start-index 1560, 17 records, no forward + no baseline):
  - CV learned (no-cycle) F1 0.823, tc F1 0.823, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1560: learned F1 0.827
- Temp‑0.8 slice (start-index 1600, 19 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.874, tc F1 0.859, tc rate 0.842
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1600: learned F1 0.878, tc F1 0.862
- Temp‑0.8 slice (start-index 1620, 20 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.789, tc F1 0.789, tc rate 0.950
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1620: learned F1 0.789, tc F1 0.789
  - transfer (train 128-cand start 1580+1600+1640, no-cycle) → test start 1620: learned F1 0.787, tc F1 0.787
- Temp‑0.8 slice (start-index 1640, 20 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.862, tc F1 0.869, tc rate 0.900
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1640: learned F1 0.870, tc F1 0.869
  - transfer (train 128-cand start 1580+1600+1620) → test start 1640: learned F1 0.870, tc F1 0.869
- Temp‑0.8 slice (start-index 1660, 19 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.839, tc F1 0.850, tc rate 0.947
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1660: learned F1 0.855, tc F1 0.855
- Temp‑0.8 slice (start-index 1680, 17 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.868, tc F1 0.868, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1680: learned F1 0.855, tc F1 0.855
- Temp‑0.8 slice (start-index 1700, 19 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.821, tc F1 0.821, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1700: learned F1 0.804, tc F1 0.804
- Temp‑0.8 slice (start-index 1720, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.849, tc F1 0.852, tc rate 0.944
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1720: learned F1 0.853, tc F1 0.853
- Temp‑0.8 slice (start-index 1740, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.824, tc F1 0.813, tc rate 0.833
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1740: learned F1 0.831, tc F1 0.819
- Temp‑0.8 slice (start-index 1760, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.841, tc F1 0.836, tc rate 0.944
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1760: learned F1 0.836, tc F1 0.836
- Temp‑0.8 slice (start-index 1780, 20 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.878, tc F1 0.878, tc rate 0.950
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1780: learned F1 0.880, tc F1 0.880
- Temp‑0.8 slice (start-index 1800, 19 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.865, tc F1 0.874, tc rate 0.947
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1800: learned F1 0.873, tc F1 0.873
- Temp‑0.8 slice (start-index 1820, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.839, tc F1 0.793, tc rate 0.556
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1820: learned F1 0.848, tc F1 0.797
- Temp‑0.8 slice (start-index 1840, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.847, tc F1 0.847, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1840: learned F1 0.853, tc F1 0.853
- Temp‑0.8 slice (start-index 1860, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.820, tc F1 0.820, tc rate 1.000
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1860: learned F1 0.812, tc F1 0.812
- Temp‑0.8 slice (start-index 1880, 20 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.873, tc F1 0.885, tc rate 0.950
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1880: learned F1 0.885, tc F1 0.885
- Temp‑0.8 slice (start-index 1900, 20 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.836, tc F1 0.856, tc rate 0.950
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1900: learned F1 0.818, tc F1 0.837
- Temp‑0.8 slice (start-index 1920, 18 records, 128 candidates, no forward + no baseline):
  - CV learned (no-cycle) F1 0.815, tc F1 0.784, tc rate 0.222
  - transfer (train temp‑0.8 merged 740–1460, no-cycle) → test start 1920: learned F1 0.834, tc F1 0.784
- Mixed‑temperature training (Kimina‑7B forward, no baseline feature):
  - train temp‑0.6 + temp‑0.8(<1100) → test temp‑0.8(>=1100): learned F1 0.822 (slightly below temp‑0.8‑only training at 0.823)
  - train temp‑0.6 + temp‑0.8(<1100) → test temp‑0.6: learned F1 0.817 (slightly above temp‑0.8‑only training at 0.812)
- Temp‑0.6 merged (start-index 600–700, 70 records, Kimina‑7B forward scores):
  - baseline F1 0.765; consensus F1 0.808; learned F1 0.814 (forward+baseline features)
  - learned reranker beats consensus on this split as well
- Temp‑0.6 merged (start-index 600–700, 70 records, Kimina‑7B forward, no baseline feature):
  - learned F1 0.816, tc F1 0.819 (slightly better than baseline-feature model)
- Temp‑0.6 merged (start-index 600–700, 70 records, no forward + no baseline features):
  - learned F1 0.826, tc F1 0.823 (best so far; forward feature not needed)

### Forward score correlations

- Candidate-level Pearson correlation on merged temp‑0.8 (9027 candidates):
  - Qwen2.5‑1.5B forward: corr(forward, F1) = 0.204; corr(forward, cycle) = 0.248
  - Qwen2.5‑3B forward: corr(forward, F1) = 0.266; corr(forward, cycle) = 0.285
  - Kimina‑7B forward: corr(forward, F1) = 0.491; corr(forward, cycle) = 0.232
  - Qwen2.5‑7B forward: corr(forward, F1) = 0.303; corr(forward, cycle) = 0.291
- Temp‑0.6 merged (Kimina‑7B forward, 1468 candidates):
  - corr(forward, F1) = 0.437; corr(forward, cycle) = 0.150

### Forward top‑k consensus

- New script: `analysis_forward_consensus_k_sweep.py` (take top‑k by forward score, then pick consensus).
- Temp‑0.8 merged (Kimina‑7B forward):
  - all‑candidates best k=16 → F1 0.821 (count 165), similar to full consensus (0.822)
  - tc‑only best k=32 → F1 0.839 but only 91 records (not directly comparable)
- Temp‑0.8 merged (Qwen2.5‑7B forward):
  - all‑candidates best k=32 → F1 0.819 (count 153), below full consensus
- Temp‑0.6 merged (Kimina‑7B forward):
  - all‑candidates best k=16 → F1 0.822 (count 43), but higher k gains rely on fewer records

### Forward margin gating (ad‑hoc)

- On Kimina‑7B forward merged set, gating between forward‑top1 and full consensus by forward margin never beats consensus.
- Best threshold is effectively “always consensus”: F1 0.822, tc 0.767.

### Consensus+cycle sweep (merged)

- Temp‑0.8 merged (start-index 740–1460): best all‑F1 0.811 at alpha 0.0 beta 3.0 (below consensus 0.822).
- Temp‑0.6 merged (start-index 600–700): best all‑F1 0.806 at alpha 0.0 beta 2.0 (below consensus 0.808).

### Consensus+cycle sweep (per-slice)

- Temp‑0.8 64-candidate slices (start-index 740–1560, 42 slices):
  - best consensus+cycle beats consensus in 14/42 slices; average delta -0.0027.
  - weighted F1: consensus 0.822 → best consensus+cycle 0.819 (overall drop -0.0027).
  - largest gains: start 1180 (+0.033, alpha 0.002 beta 2.0), start 1140 (+0.029, alpha 0.0 beta 2.0), start 1560 (+0.027, alpha 0.0 beta 1.0).
  - most common best alpha/beta: 0.0/3.0 (13 slices), 0.002/3.0 (5), 0.005/3.0 (5).

### Learned reranker feature ablations (no forward, no baseline)

- Added feature-drop flags to `analysis_learned_reranker.py`: `--no-cycle`, `--no-length`, `--no-consensus`, `--no-typecheck`.
- Temp‑0.8 merged (start-index 740–1460):
  - drop cycle → learned F1 0.830 (best; cycle not needed)
  - drop length → learned F1 0.825
  - drop consensus → learned F1 0.759 (big drop)
  - drop typecheck → learned F1 0.825 but tc rate falls to 0.727
- Temp‑0.6 merged (start-index 600–700):
  - drop cycle → learned F1 0.806 (hurts vs 0.826)
  - drop length → learned F1 0.822 (hurts vs 0.826)
  - drop consensus → learned F1 0.772 (big drop)

### Non-linear reranker (GBRT)

- Added `--model gbrt` option to `analysis_learned_reranker.py` (GradientBoostingRegressor).
- Temp‑0.8 merged (start-index 740–1460, default features):
  - learned F1 0.826, tc F1 0.821; below ridge (0.830).
  - avg feature importances: cycle 0.061, length 0.192, consensus 0.624, typecheck 0.124.
- Temp‑0.8 merged (no-cycle):
  - learned F1 0.815, tc F1 0.811; worse than ridge no-cycle (0.830).
  - avg feature importances: length 0.229, consensus 0.644, typecheck 0.126.
- Temp‑0.6 merged (start-index 600–700, default features):
  - learned F1 0.794, tc F1 0.798; below ridge (0.826).

### Embedding consensus (candidate similarity)

- New script: `analysis_embedding_consensus.py` (embedding-based consensus via cosine similarity).
- Temp‑0.8 merged 128‑cand (start-index 1580–2260, 668 records):
  - MiniLM (`sentence-transformers/all-MiniLM-L6-v2`): embed F1 0.829, tc F1 0.829; below token-consensus 0.835.
  - CodeBERT (`microsoft/codebert-base`): embed F1 0.830, tc F1 0.830; still below token-consensus.
  - UniXCoder (`microsoft/unixcoder-base`): embed F1 0.828, tc F1 0.827; below token-consensus.
- Temp‑0.8 merged 64‑cand (start-index 740–1460, 172 records):
  - CodeBERT embed F1 0.817, tc F1 0.819; below token-consensus 0.822.
- CodeBERT embed vs token-consensus (per-record F1 deltas):
  - 128‑cand 1580–2260: wins 196, losses 239, ties 233; avg delta -0.0067.
  - 64‑cand 740–1460: wins 44, losses 59, ties 69; avg delta -0.0056.

### Temp‑0.8 slice start-index 2280 (128 candidates)

- Generated 3 shards (Qwen2.5‑1.5B cycle model), merged 19 records.
- Rescored cycle on GPU (file suffix `_rescored.json`).
- Statement similarity:
  - baseline F1 0.822; best‑cycle F1 0.730; best‑cycle‑tc F1 0.795.
- Consensus rerank:
  - baseline F1 0.822 → consensus F1 0.843, tc F1 0.829; consensus tc rate 1.0.
  - tc‑only consensus F1 0.818; tc better in 10.5% of records.
- Consensus k‑sweep:
  - all‑candidates best k=64 → F1 0.853 (count 480).
  - tc‑only best k=64 → F1 0.838 (count 390).
- Consensus+cycle sweep:
  - best all‑F1 at alpha 0.01 beta 3.0 → F1 0.846, tc rate 0.632.
- Oracle:
  - oracle F1 0.949, oracle‑tc F1 0.940 (headroom remains).
- Cycle correlation:
  - avg Pearson 0.381, Spearman 0.259.
- Transfer gate (train 1580–2260 → test 2280):
  - base gate F1 0.836 (below consensus 0.843).
  - extra‑stats gate F1 0.847 (beats consensus), choose‑tc rate 0.474.
- Transfer learned reranker (train 1580–2260 → test 2280):
  - default features F1 0.846, tc F1 0.840 (slightly above consensus).
  - no‑cycle features F1 0.841.
  - extra candidate features: F1 0.837 (no gain).
  - extra candidate features + no‑cycle: F1 0.837 (no gain).

### Temp‑0.8 merged start-index 1580–2280 (128 candidates, 687 records)

- Merged 1580–2260 (CPU‑rescored) + 2280 (GPU‑rescored) into `kimina_bestof_cycle_lenpen001_128_20_temp08_merged_start1580_2280_rescored.json`.
- Statement similarity:
  - baseline F1 0.778; best‑cycle F1 0.738; best‑cycle‑tc F1 0.774.
- Consensus rerank:
  - consensus F1 0.836, tc F1 0.835 (baseline 0.778); tc rate 0.996.
- Consensus tc‑compare:
  - tc‑only consensus F1 0.833; tc better in 17.4% of records.
- Oracle:
  - oracle F1 0.935, oracle‑tc F1 0.920.
- Typecheck signal:
  - candidate tc rate 0.669; tc‑candidate F1 0.786; non‑tc 0.719.
- Cycle correlation:
  - avg Pearson 0.320, Spearman 0.179.
- Consensus k‑sweep (reduced trials=5, ks=8/16/32/64):
  - all‑candidates best k=32 → F1 0.833 (tc 0.824).
  - tc‑only best k=32 → F1 0.841 (count 2715), but fewer records.
- New script: `analysis_consensus_k_sweep_fast.py` (cached pairwise F1).
- Consensus k‑sweep (trials=30, ks=4/8/16/32/64):
  - all‑candidates best k=32 → F1 0.832 (tc 0.820).
  - tc‑only best k=32 → F1 0.840 (count 16290), but fewer records.
- NL↔Lean embedding rerank (new script `analysis_nl_embedding_rerank.py`, E5‑base bi‑encoder):
  - merged 687‑record set: embed F1 0.713, tc F1 0.750 (well below baseline).
  - deduped 173‑record set: embed F1 0.696, tc F1 0.738 (well below baseline).
- NL↔Lean cross‑encoder rerank (new script `analysis_nl_cross_encoder_rerank.py`, ms‑marco MiniLM):
  - merged 687‑record set: cross F1 0.705, tc F1 0.735 (well below baseline).
  - deduped 173‑record set: cross F1 0.688, tc F1 0.735 (well below baseline).
- Meta‑selector between consensus and learned reranker (new script `analysis_meta_selector.py`):
  - merged 687‑record set: meta F1 0.840 (consensus 0.836, learned 0.841), choose‑learned 0.77.
  - deduped 173‑record set: meta F1 0.840 (consensus 0.832, learned 0.840), choose‑learned 0.58.
  - v2 meta features (add learned score std): merged F1 0.8404; deduped F1 0.8400 (still below learned).
  - GBRT meta model: merged F1 0.8407 (still below learned), deduped F1 0.835 (worse).
- New script: `analysis_consensus_cycle_sweep_fast.py` (cached consensus stats).
  - best all‑F1 at alpha 0.0 beta 3.0 → F1 0.826 (below consensus 0.836).
- Learned reranker (5‑fold CV):
  - default features F1 0.841, tc F1 0.839 (beats consensus).
  - no‑cycle F1 0.840 (slightly worse).
- Learned reranker + embedding consensus feature:
  - MiniLM (`sentence-transformers/all-MiniLM-L6-v2`): F1 0.8405, tc F1 0.8388 (no gain).
  - CodeBERT (`microsoft/codebert-base`): F1 0.8399, tc F1 0.8378 (worse).
  - E5-base (`intfloat/e5-base-v2`): F1 0.8410, tc F1 0.8394 (no gain).
- Learned reranker + context interactions (tc_rate/consensus margin/avg pairwise):
  - ridge: F1 0.8402, tc F1 0.8405 (no gain).
  - gbrt: F1 0.8260 (worse).
- Learned reranker + extra candidate features (lenpen z, ranks, tc‑consensus):
  - merged ridge: F1 0.8431, tc F1 0.8383 (new best); typecheck 0.977.
  - merged ridge no‑cycle: F1 0.8436, tc F1 0.8394 (best so far).
  - merged gbrt: F1 0.8349 (worse).
  - dedup ridge: F1 0.8396, tc F1 0.8381 (still below merged best).
  - dedup ridge no‑cycle: F1 0.8396, tc F1 0.8380.
  - slice‑CV (1580–2280, 36 slices): no‑cycle F1 0.8397 → extra‑candidate no‑cycle F1 0.8434.
  - per‑slice wins: extra‑candidate no‑cycle beats no‑cycle in 21/36 slices.
  - Cross‑domain test: model trained on 128‑cand (extra‑candidate no‑cycle) applied to 64‑cand merged improves slightly (F1 0.8305 vs low‑model 0.8294).
  - Mixed gate by candidate count (new `analysis_reranker_mixed_gate.py`) always prefers the 128‑trained model; best threshold is 0.
  - 64‑candidate merged (start 740–1460): extra‑candidate no‑cycle F1 0.8247 (worse than no‑cycle 0.8296).
  - Combined 64+128 merged (859 records): no‑cycle F1 0.8379 → extra‑candidate no‑cycle F1 0.8407 (CV).
  - Transfer (train 64‑cand → test 128‑cand, extra‑candidate no‑cycle): F1 0.8426 (below 128‑trained best 0.8436).
- Learned tc‑gate (5‑fold CV, tuned threshold):
  - base features gate F1 0.842 (beats consensus).
  - extra‑stats gate F1 0.840 (slightly worse).
- Deduped by id (173 records) on the same merged file:
  - baseline F1 0.767; consensus F1 0.832; tc‑consensus F1 0.833.
  - learned reranker F1 0.840, tc F1 0.841 (still best).
  - oracle F1 0.933, tc F1 0.917 (headroom persists).
  - learned tc‑gate (deduped): base gate F1 0.835, extra‑stats gate F1 0.834 (both slightly below tc‑consensus).
  - consensus k‑sweep (fast, trials=30): all‑candidates best k=32 → F1 0.831; tc‑only best k=32 → F1 0.847 (fewer records).

### Structural consensus + forward rerank pilot

- New script: `analysis_consensus_structural.py` (identifier masking + digit-aware tokenization).
  - Bug fix: corrected regex escapes for `\d` and `\s`; earlier structural-consensus numbers were invalid.
- Structural consensus with corrected tokenization (mask=none, eval with standard F1):
  - combined 64+128 (859): consensus F1 0.833 (matches standard 0.833); tc F1 0.832.
  - merged 128 (687): consensus F1 0.836 (matches standard 0.836); tc F1 0.835.
  - dedup 128 (173): consensus F1 0.832; tc F1 0.836 (tc slightly above standard 0.833).
  - merged 64 (172): consensus F1 0.822 (matches standard 0.822).
  - Divergence vs standard consensus is rare (1–2% of records) and F1 is unchanged.
  - Diff records are all numeric statements (17/17 on combined); no F1 gains (0 better, 1 worse).
- Numeric statements are harder for consensus after stripping theorem names:
  - combined 64+128: digits (546) → consensus F1 0.828, improve rate 0.581; no digits (313) → consensus F1 0.842, improve rate 0.684.
  - merged 128: digits (447) → consensus F1 0.829, improve rate 0.568; no digits (240) → consensus F1 0.848, improve rate 0.675.
- Identifier masking (short/all) degrades consensus (combined 64+128: F1 0.828).
- Number masking (`--mask-numbers`) does not help (combined 64+128: F1 0.833 → 0.8326).
- Forward rerank updates:
  - `load_hf_causal_lm` now retries GPU float16 if bf16 logits are degenerate (fixes Qwen2.5 GPU loads).
  - `analysis_forward_rerank.py` supports `--score-top-k` and `--score-top-k-by` to cap candidate scoring.
  - `analysis_forward_rerank.py` now trims output records when `--max-records` is set.
- Pilot forward scoring (deduped 128, first 20 records, top-32 by cycle, Qwen2.5-3B-Instruct):
  - best alpha=0.0 beta=1.0: all_f1 0.776; tc_f1 0.780.
  - baseline F1 on same 20 records: 0.808 (forward rerank does not beat baseline here).
  - 50-record run timed out due to CPU fallback after degenerate GPU logits.
- Learned reranker interpretability:
  - `analysis_learned_reranker.py` now emits `feature_names`.
  - Combined 64+128 extra-candidate no-cycle weights: cons_z (0.055) and typecheck (0.043) dominate; tc_cons_z/cons_rank (~0.021) next; length and rank features are small (~0.01 or less).
- Numeric feature probes (combined 64+128, extra-candidate no-cycle):
  - candidate numeric features hurt overall F1 (0.8407 → 0.8389).
  - new NL-number overlap feature (`--nl-number-features`) slightly improves overall F1 to 0.8410.
  - digits subset improves (0.83916 → 0.83982), no-digit subset slightly down (0.84325 → 0.84315).
  - detailed NL-number features (`--nl-number-detail`) hurt overall (F1 0.8396).
  - new script `analysis_reranker_digit_gate.py` to gate base vs nl-number models.
  - gate by NL digits: F1 0.84094 (slightly below nl-number 0.84104).
  - gate by GT digits: F1 0.84107 (tiny gain; uses unavailable signal at inference).
- NL-number heuristic for consensus (new `analysis_consensus_nl_numbers.py`):
  - combined 64+128: NL-number filter hurts badly (consensus 0.833 → 0.807).
  - digits subset: consensus 0.826 → 0.784 (fails); no-digit subset unchanged (falls back to consensus).
  - added word-number extraction (e.g., "two") to NL parsing:
    - `--nl-number-features` now reaches F1 0.8413 (best combined so far), improves numeric subset (0.8369 → 0.8376).
    - merged 128 set: extra-candidate no-cycle + NL numbers reaches F1 0.8445 (new best vs 0.8436).
    - slice-CV (1580–2260, 128): extra-candidate no-cycle + NL numbers F1 0.8441 (vs 0.8434).
    - digit-gate by NL numbers gives F1 0.84115 (still below nl-number model).
    - NL-number consensus heuristic still worse (overall F1 0.799; digits 0.775).
    - precision/recall metrics also underperform (overall F1 0.800/0.807; digits 0.776/0.787).
    - selection modes:
      - perfect match: identical to consensus (no effect).
      - no-missing: F1 0.815 (worse; digits 0.800).
      - no-extra: F1 0.832 (slightly below consensus).
    - consensus number-penalty sweep (new `analysis_consensus_number_penalty.py`):
      - any missing/extra penalty lowers F1; best is alpha=beta=0 (plain consensus).
    - merged 64 set: extra-candidate no-cycle + NL numbers F1 0.823 (still below previous 0.8247).
    - dedup 128 set: extra-candidate no-cycle + NL numbers F1 0.8387 (below 0.8396).
    - missing/extra count features (`--nl-number-missing-extra`) give a slightly better combined score:
      - combined 64+128: F1 0.84135 (best combined so far), tc F1 0.83679.
      - adding nl_num_f1 to missing/extra drops to F1 0.84122.
      - adding missing/extra ratios does not improve (F1 0.84134).
      - slice-CV 128 with ratios: F1 0.84393 (below 0.84413).
      - adding context interactions degrades (F1 0.8378).
      - signed delta alone: F1 0.84072 (worse than missing/extra).
      - missing/extra + signed delta: F1 0.84138 (tiny gain vs 0.84135); slice-CV 128 F1 0.84411.
      - signed delta alone on merged 128: F1 0.84352 (below 0.8445).
      - digit-gate with missing/extra + signed delta: F1 0.84050 (worse).
      - transfer 128 -> 64 (missing/extra + signed delta): F1 0.83158 (same as missing/extra).
      - transfer 64 -> 128 (missing/extra + signed delta): F1 0.84334 (slightly above 0.84326).
      - merged 128: F1 0.84448 (about equal to 0.8445).
      - slice-CV 128: F1 0.84413 (slightly above 0.84410).
      - merged 64: F1 0.82317 (still below 0.8247).
      - transfer 128 -> 64: F1 0.83158 (slightly above 0.8305).
      - transfer 64 -> 128: F1 0.84326 (slightly above 0.8426).
      - numeric split (combined 64+128): numeric subset 0.83688 -> 0.83746; non-numeric 0.84766 -> 0.84855.
      - dedup 128: F1 0.8385 (still below 0.8396/0.8387).
      - digit-gate with missing/extra still below NL model (F1 0.84095).
- Tests (LeanInteract):
  - `pytest -q LeanInteract/tests/test_utils.py`: 14 passed.
  - `pytest -q LeanInteract/tests/test_git_functionality.py`: 8 passed.
  - `pytest -q LeanInteract/tests/test_concurrency.py`: 13 passed.
  - `pytest -q LeanInteract/tests/test_server.py`: 48 passed in 93s.
- Gate classifier + NL rerank probes:
  - `analysis_reranker_gate_classifier.py` now caches per-record features to avoid repeated `_prepare_record` calls; baseline F1 now uses baseline candidate.
  - merged 64+128 (rescored): base F1 0.84065, nl-missing/extra F1 0.84135.
  - 5-fold gate classifier chooses NL 0% of the time; gate F1 0.84065 (no gain).
  - NL-missing/extra beats base on only 5.1% of records.
  - heuristic gate on NL-number presence picks NL on 61.9% and yields F1 0.84104 (below always-NL 0.84135).
- Cross-encoder rerank (NL↔Lean):
  - `analysis_nl_cross_encoder_rerank.py` with `cross-encoder/ms-marco-MiniLM-L-6-v2` on merged 64+128 (rescored) is worse:
    - cross F1 0.705, cross-tc F1 0.734, typecheck rate 0.464.
    - baseline F1 0.774 on the same records.
- Embedding rerank (NL↔Lean):
  - `analysis_nl_embedding_rerank.py` with `intfloat/e5-base-v2` on merged 64+128 (rescored) is worse:
    - embed F1 0.713, embed-tc F1 0.750, typecheck rate 0.480.
    - baseline F1 0.774 on the same records.
- Learned reranker with embed-consensus:
  - merged 64+128 (rescored), extra-candidate no-cycle + NL missing/extra + embed consensus:
    - learned F1 0.841434, learned-tc F1 0.836594 (slightly above 0.84135 baseline).
    - `embed_cons_z` weight is small (~0.0053) but net performance bumps a bit.
- Tests (LeanInteract full suite):
  - `pytest -q` in `LeanInteract/`: 83 passed in 167s.
