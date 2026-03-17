# ICCAD Research Folder

This folder contains research notes, experiment results, and documentation for the ICCAD project.

## Research Goals

1. **Activation-based pruning**: per-layer `top_k` reduction for MoE models, guided by activation statistics.
2. **Expert quantization**: per-layer mixed-precision quantization for MoE expert weights, guided by importance metrics.

Both techniques are developed and extended from `tools/imatrix/`. Target models: Qwen3 MoE and Qwen3.5 MoE series.

## Structure

```
iccad/
├── README.md       - this file
├── notes/          - reading notes, code understanding, and ideas
└── results/        - experiment outputs, perplexity scores, imatrix stats
```
