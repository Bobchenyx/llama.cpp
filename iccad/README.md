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
| [results/ppl-perlayer-topk.md](results/ppl-perlayer-topk.md) | Per-layer top_k PPL validation for all 4 models (30B, 35B, 122B, 235B). CKA M1 rankings + code modifications |
| [results/cka-analysis.md](results/cka-analysis.md) | CKA methodology: bootstrap, token vs chunk, probe point comparison, large model results |
| [results/router-analysis.md](results/router-analysis.md) | Router metrics (score margin, weight concentration) for Qwen3-30B, Qwen3.5-35B, Qwen3.5-122B |

## Data

- Model files: `model/qwen3/`, `model/qwen35/`
- CKA probe data: `data/qwen3-235b-hsic/`, `data/qwen35-122b-hsic/`, `data/qwen3-30b-hsic-lout/`, `data/qwen35-35b-hsic-lout/`
- imatrix outputs: `iccad/results/` (`.txt` files)
- Schedule files: `iccad/schedules/`, `iccad/schedules/qwen35/`, `iccad/schedules/qwen35-122b/`, `iccad/schedules/qwen3-235b/`
- Analysis scripts: `iccad/scripts/`
