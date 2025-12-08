# autoformalize

## Kimina scripts

- `run_kimina.py` – single-problem demo that streams one Kimina generation for a math word problem.
- `run_kimina_beq_plus.py` – baseline BEq+ evaluation without SMC, mirroring the ProofNetVerif setup.
- `run_kimina_genlm_full_pipeline.py` – full GenLM-controlled pipeline with Lean well-typed and cycle-consistency potentials, posterior scoring, and BEq+ evaluation/summary reporting. It now relies only on `AI-MO/Kimina-Autoformalizer-7B` as the primary generator and `Qwen/Qwen2.5-7B-Instruct` as the reference LM for the potentials, so no massive 70B checkpoints are required. Use `--smc-config full` (or `--no-ablations`) to run only the main SMC configuration from the paper, or `--smc-config all` to include the slower ablations.

Example full-pipeline run:

```bash
python run_kimina_genlm_full_pipeline.py \
  --n-examples 10 \
  --primary-model-id AI-MO/Kimina-Autoformalizer-7B \
  --reference-model-id Qwen/Qwen2.5-7B-Instruct \
  --max-tokens 256 \
  --num-particles 3 \
  --smc-config full
```

`full` runs the complete SMC method (Lean + cycle potentials) described in the paper, while `all` also launches the individual ablations and therefore runs much longer. You can further dial down runtime with `--num-particles`, `--max-tokens`, `--potential-stride`, or (if bitsandbytes is installed) `--four-bit` to quantize both models.

Install dependencies such as `genlm-control`, `datasets`, `lean-interact`, and the HF checkpoints for the primary (Kimina) and reference (default `Qwen/Qwen2.5-7B-Instruct`) models so that the scripts can load Kimina, steer with GenLM, and interact with Lean.
