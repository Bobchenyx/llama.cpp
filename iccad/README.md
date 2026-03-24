# ICCAD Research Folder

This folder contains research notes, experiment results, and documentation for the ICCAD project.

## Research Goals

1. **Activation-based pruning**: per-layer `top_k` reduction for MoE models, guided by activation statistics.
2. **Expert quantization**: per-layer mixed-precision quantization for MoE expert weights, guided by importance metrics.

Both techniques are developed and extended in `tools/imatrix-iccad/` (our working copy). `tools/imatrix/` is kept as a fixed bug-fixed upstream baseline. Target models: Qwen3 MoE and Qwen3.5 MoE series.

## Structure

```
iccad/
├── README.md       - this file (index of all documents)
├── notes/          - reading notes, code understanding, and ideas
└── results/        - experiment outputs, perplexity scores, imatrix stats
```

## Notes Index

| File | Content |
|------|---------|
| [notes/imatrix-code-analysis.md](notes/imatrix-code-analysis.md) | Complete walkthrough of `tools/imatrix/imatrix.cpp`: data flow, collection logic for dense vs MoE layers, statistics, file formats, and extension points. Also documents ICCAD extensions implemented in `tools/imatrix-iccad/` (per-expert activation count display). |
| [notes/expert-activation-analysis.md](notes/expert-activation-analysis.md) | Analysis of per-expert activation distributions for Qwen3 MoE under E8 vs E4 settings. Key findings: ffn_down sensitivity increases with depth; Gini and ΔEntropy as pruning-safety metrics; preliminary layer groupings for mixed top_k schedule. |
| [notes/per-layer-sensitivity-design.md](notes/per-layer-sensitivity-design.md) | Design document for per-layer top_k sensitivity analysis. Three approaches compared: offline replay (recommended), per-layer inference, router logit capture. Addresses distribution shift problem in pure E8/E4 comparison. |

## Results Index

| Description | Files | Date |
|-------------|-------|------|
| Qwen3 MoE imatrix statistics (E8, q8_0 calibration) | `qwen3-imatrix-q8_0-statistics-iccad.txt`, `expert_activation_counts.png` | 2026-03-17 |
| Qwen3 MoE imatrix statistics (E4, q8_0 calibration) | `qwen3-imatrix-q8_0-E4-statistics-iccad.txt`, `expert_activation_counts_E4.png` | 2026-03-17 |

> Result files live in the workspace root (`/home/user1/workspace/bobchenyx/26-ICCAD/`), not committed to the repo.
