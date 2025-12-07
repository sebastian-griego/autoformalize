# autoformalize

## Kimina scripts

- `run_kimina.py` – single-problem demo that streams one Kimina generation for a math word problem.
- `run_kimina_beq_plus.py` – baseline BEq+ evaluation without SMC, mirroring the ProofNetVerif setup.
- `run_kimina_genlm_full_pipeline.py` – full GenLM-controlled pipeline with Lean well-typed and cycle-consistency potentials, posterior scoring, and BEq+ evaluation/summary reporting. It now relies only on `AI-MO/Kimina-Autoformalizer-7B` as the primary generator and `Qwen/Qwen2.5-7B-Instruct` as the reference LM for the potentials, so no massive 70B checkpoints are required.

Example full-pipeline run:

```bash
python run_kimina_genlm_full_pipeline.py \
  --n-examples 2 \
  --primary-model-id AI-MO/Kimina-Autoformalizer-7B \
  --reference-model-id Qwen/Qwen2.5-7B-Instruct
```

Install dependencies such as `genlm-control`, `datasets`, `lean-interact`, and the HF checkpoints for the primary (Kimina) and reference (default `Qwen/Qwen2.5-7B-Instruct`) models so that the scripts can load Kimina, steer with GenLM, and interact with Lean.
