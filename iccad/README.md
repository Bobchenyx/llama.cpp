# ICCAD Research Folder

## Research Goals

1. **Activation-based pruning**: per-layer `top_k` reduction for MoE models, guided by CKA layer importance.
2. **Expert quantization**: per-layer mixed-precision quantization for MoE expert weights.

Tools: `tools/imatrix-iccad/` (imatrix + router analysis), `tools/imatrix-hsic/` (CKA probe).
Target models: Qwen3 MoE and Qwen3.5 MoE series.

## Documents

| File | Content |
|------|---------|
| [notes/imatrix-code-analysis.md](notes/imatrix-code-analysis.md) | imatrix tool internals, bug fixes, ICCAD extensions (per-expert counts, router analysis) |
| [results/ppl-perlayer-topk.md](results/ppl-perlayer-topk.md) | Per-layer top_k PPL validation for Qwen3-30B and Qwen3.5-35B. CKA M1 rankings + code modifications |
| [results/cka-analysis.md](results/cka-analysis.md) | CKA methodology: bootstrap stability, token-level vs chunk-averaged comparison, PPL validation |
| [results/router-analysis.md](results/router-analysis.md) | Router metrics (score margin, weight concentration) for Qwen3-30B, Qwen3.5-35B, Qwen3.5-122B |

## Data

- Model files: `qwen3/`, `qwen35/`
- CKA probe data: `qwen3/hsic/`, `qwen35/hsic/`
- imatrix outputs: `qwen3/imatrix/`, workspace root `.txt` files
- Schedule files: `iccad/schedules/`, `iccad/schedules/qwen35/`
- Analysis scripts: `iccad/scripts/`
