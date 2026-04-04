# ICCAD Research Folder

This folder contains research notes, experiment results, and documentation for the ICCAD project.

## Research Goals

1. **Activation-based pruning**: per-layer `top_k` reduction for MoE models, guided by activation statistics.
2. **Expert quantization**: per-layer mixed-precision quantization for MoE expert weights, guided by importance metrics.

Both techniques are developed and extended in `tools/imatrix-iccad/` (our working copy). `tools/imatrix/` is kept as a fixed bug-fixed upstream baseline. Target models: Qwen3 MoE and Qwen3.5 MoE series.

3. **CKA-based layer importance**: `tools/imatrix-hsic/` captures per-layer hidden representations during calibration inference. Offline CKA (Centered Kernel Alignment) computation produces S_in(i) = CKA(X, H_i) and S_out(i) = CKA(H_i, H_L), yielding a per-layer importance metric M1 = S_out × (1 - S_in).

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
| [notes/imatrix-code-analysis.md](notes/imatrix-code-analysis.md) | Complete walkthrough of `tools/imatrix/imatrix.cpp`: data flow, collection logic for dense vs MoE layers, statistics, file formats, and extension points. Documents ICCAD extensions in `tools/imatrix-iccad/`: per-expert activation count display and router-level analysis (score margin, weight concentration). |
| [notes/expert-activation-analysis.md](notes/expert-activation-analysis.md) | Analysis of per-expert activation distributions for Qwen3 MoE under E8 vs E4 settings. Key findings: ffn_down sensitivity increases with depth; Gini and ΔEntropy as pruning-safety metrics; preliminary layer groupings for mixed top_k schedule. |
| [notes/per-layer-sensitivity-design.md](notes/per-layer-sensitivity-design.md) | Design document for per-layer top_k sensitivity analysis. Three approaches compared: offline replay, per-layer inference, router logit capture. Phase 1a (router-level analysis in callback) implemented: score margin, weight concentration, max prob, routing entropy. |

## Results Index

| Description | Files | Date |
|-------------|-------|------|
| Qwen3 MoE imatrix (E8, router v2 with k4+k6) | `qwen3-imatrix-q8_0-callback-864.txt` | 2026-03-24 |
| Qwen3 MoE imatrix (E4, q8_0 calibration) | `qwen3-imatrix-q8_0-E4-statistics-iccad.txt` | 2026-03-17 |
| Qwen3 router analysis vs ΔEntropy (k=4 + k=6) | [results/router-analysis-results.md](results/router-analysis-results.md) | 2026-03-24 |
| Qwen3.5-35B router analysis + sensitivity ranking | [results/qwen35-router-analysis.md](results/qwen35-router-analysis.md), `qwen35-imatrix-q8_0-callback-864.txt` | 2026-03-24 |
| Qwen3.5-122B router analysis + cross-scale comparison | [results/qwen35-122b-router-analysis.md](results/qwen35-122b-router-analysis.md), `qwen35-122B-imatrix-q8_0-callback-864.txt` | 2026-03-25 |
| Qwen3-30B CKA/HSIC layer representations (840 chunks) | Raw data in `qwen3/hsic/` (51 files, ~340 MB) | 2026-04-04 |

> Model files live in `qwen3/` and `qwen35/`. Analysis scripts in `scripts/`. imatrix outputs at workspace root or in `qwen3/imatrix/`. HSIC probe data in `qwen3/hsic/`.
