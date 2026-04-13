# Per-Layer top_k Perplexity Validation

## Setup

- Benchmark: wikitext-2-raw (wiki.test.raw), ctx=512, 584 chunks
- Mechanism: per-layer `n_expert_used` schedule via `--override-kv expert_schedule=str:<path>`
- Schedule format: `layer_id top_k` per line; unlisted layers keep model default (k=8)
- Last MoE layer always kept at k=8
- Keep ratio: ~25% of layers at k=8, rest at k=4

---

## Qwen3-30B (48 MoE layers, 128 experts, default k=8)

35 layers at k=4, 13 at k=8 (12 selected + layer 47).

| Run | Setting | 12 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 48 | 7.3598 ± 0.052 | — |
| A3  | **token-level CKA M1 top-12** | [4,8,34,37,38,39,40,41,42,43,44,45] | **7.9370 ± 0.056** | +0.577 |
| A1  | chunk-averaged CKA M1 top-12 | [3,6,8,10,12,37,39,42,43,44,45,46] | 7.9905 ± 0.057 | +0.631 |
| A2  | chunk-averaged CKA M2 top-12 | [0,8,10,11,12,37,39,42,43,44,45,46] | 7.9912 ± 0.057 | +0.631 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36,40,44] | 8.1357 ± 0.058 | +0.776 |
| B   | chunk-averaged CKA M1 bottom-12 | [0,5,9,19,21,26,28,31,32,33,34,38] | 8.2323 ± 0.059 | +0.873 |
| B2  | all k=4 | none | 8.6199 ± 0.062 | +1.260 |

---

## Qwen3.5-35B (40 MoE layers, 256 experts + shared experts, default k=8)

29 layers at k=4, 11 at k=8 (10 selected + layer 39).

| Run | Setting | 10 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 40 | 6.5359 ± 0.042 | — |
| A2  | **token-level CKA M1 top-10** | [19,29,31,32,33,34,35,36,37,38] | **7.0799 ± 0.046** | +0.544 |
| A1  | chunk-averaged CKA M1 top-10 | [9,19,23,25,26,34,35,36,37,38] | 7.1060 ± 0.046 | +0.570 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36] | 7.2591 ± 0.047 | +0.723 |
| B   | CKA M1 bottom-10 | [0,4,5,6,8,10,12,15,16,20] | 7.3529 ± 0.049 | +0.817 |
| B2  | all k=4 | none | 7.5784 ± 0.050 | +1.043 |

---

## Qwen3.5-122B (48 MoE layers, 256 experts + shared experts, default k=8)

35 layers at k=4, 13 at k=8 (12 selected + layer 47).

| Run | Setting | 12 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 48 | 4.8192 ± 0.028 | — |
| A1  | **chunk-averaged CKA M1 top-12** | [31,35,36,37,38,39,40,42,43,44,45,46] | **5.1318 ± 0.031** | +0.313 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36,40,44] | 5.4256 ± 0.033 | +0.606 |
| B   | CKA M1 bottom-12 | [3,4,7,8,9,12,14,16,20,22,27,28] | 5.4910 ± 0.033 | +0.672 |
| B2  | all k=4 | none | 5.5646 ± 0.034 | +0.745 |

---

## Qwen3-235B (94 MoE layers, 128 experts, default k=8)

69 layers at k=4, 25 at k=8 (24 selected + layer 93).

| Run | Setting | 24 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 94 | 4.3142 ± 0.026 | — |
| A1  | **chunk-averaged CKA M1 top-24** | [0,7,18,19,21,23,25,26,27,39,41,46,53,60,61,79,84,85,86,87,89,90,91,92] | **5.0520 ± 0.031** | +0.738 |
| C   | uniform (il%4==0) | [0,4,8,...,88,92] (24 layers) | 5.1178 ± 0.031 | +0.804 |
| B   | CKA M1 bottom-24 | [2,3,12,13,14,29,31,32,33,34,35,44,48,63,65,66,67,68,69,70,71,73,74,76] | 5.2322 ± 0.032 | +0.918 |
| B2  | all k=4 | none | 5.4729 ± 0.033 | +1.159 |

---

## Analysis

**Ordering** (consistent across all 4 models):
```
all-k8  <  CKA-top  <  uniform  <  bottom-N  <  all-k4
```

1. **CKA ranking validated at scale**: CKA M1 top-N beats uniform in all 4 models.
   The advantage ranges from 0.07 PPL (235B) to 0.29 PPL (122B).

2. **Token-level CKA is best** (small models): Token-level M1 outperforms chunk-averaged
   by 0.03-0.05 PPL. Both methods are valid; token-level recommended when compute allows.

3. **M1 ≈ M2**: Two CKA metrics give identical PPL (Δ=0.0007 on 30B). M1 preferred.

4. **PPL recovery**: CKA-guided selection recovers 36-58% of the PPL gap while pruning
   ~75% of layers to k=4.

5. **Shared experts buffer Qwen3.5**: Total PPL gap is smaller for Qwen3.5 models
   (0.75-1.04) vs Qwen3 models (1.16-1.26) despite more experts.

---

## Cross-Model Comparison

| Metric | Qwen3-30B | Qwen3.5-35B | Qwen3.5-122B | Qwen3-235B |
|--------|-----------|-------------|--------------|------------|
| Layers | 48 | 40 | 48 | 94 |
| Baseline PPL | 7.3598 | 6.5359 | 4.8192 | 4.3142 |
| All k=4 PPL | 8.6199 | 7.5784 | 5.5646 | 5.4729 |
| PPL gap | 1.2601 | 1.0425 | 0.7454 | 1.1587 |
| Best CKA PPL | **7.9370** | **7.0799** | **5.1318** | **5.0520** |
| CKA method | token-level | token-level | chunk-avg | chunk-avg |
| CKA ΔPPL | +0.577 | +0.544 | +0.313 | +0.738 |
| Uniform ΔPPL | +0.776 | +0.723 | +0.606 | +0.804 |
| CKA vs uniform | -0.199 | -0.179 | -0.294 | -0.066 |
| CKA recovery % | 54.0% | 47.8% | 58.0% | 36.3% |
| Keep ratio | 12/48=25% | 10/40=25% | 12/48=25% | 24/94≈26% |

---

## Code Modifications (4 changes)

| # | File | Change |
|---|------|--------|
| 1 | `src/llama-hparams.h:49` | Add `n_expert_used_arr[LLAMA_MAX_LAYERS]` per-layer array |
| 2 | `src/llama-model.cpp:379-406` | Init array from global default + load schedule file via `--override-kv expert_schedule=str:<path>` |
| 3 | `src/models/qwen3moe.cpp:95`, `qwen35moe.cpp:379` | Use per-layer value with warmup guard: `cparams.warmup ? n_expert_used : hparams.n_expert_used_arr[il]` |
| 4 | `src/llama-graph.cpp:1527` | **Bug fix**: aggregation loop used global `hparams.n_expert_used` for iteration count, causing OOB reads when per-layer k < global k. Fixed to `min(n_expert_used, hparams.n_expert_used)` |

---

## Schedule Files

```
iccad/schedules/
├── token_m1_keep_top12.txt         # Qwen3-30B: token-level CKA M1 top-12 at k=8
├── m1_keep_top12.txt               # Qwen3-30B: chunk-averaged M1 top-12 at k=8
├── m2_keep_top12.txt               # Qwen3-30B: chunk-averaged M2 top-12 at k=8
├── m1_keep_bot12.txt               # Qwen3-30B: M1 bottom-12 at k=8
├── uniform_keep_12.txt             # Qwen3-30B: layer%4==0 at k=8
├── qwen35/
│   ├── token_m1_keep_top10.txt     # Qwen3.5-35B: token-level M1 top-10 at k=8
│   ├── m1_keep_top10.txt           # Qwen3.5-35B: chunk-averaged M1 top-10 at k=8
│   ├── m1_keep_bot10.txt           # Qwen3.5-35B: M1 bottom-10 at k=8
│   └── uniform_keep_10.txt         # Qwen3.5-35B: layer%4==0 at k=8
├── qwen35-122b/
│   ├── m1_keep_top12.txt           # Qwen3.5-122B: chunk-averaged M1 top-12 at k=8
│   ├── m1_keep_bot12.txt           # Qwen3.5-122B: M1 bottom-12 at k=8
│   ├── uniform_keep_12.txt         # Qwen3.5-122B: layer%4==0 at k=8
│   └── all_k4.txt                  # Qwen3.5-122B: all layers at k=4
└── qwen3-235b/
    ├── m1_keep_top24.txt           # Qwen3-235B: chunk-averaged M1 top-24 at k=8
    ├── m1_keep_bot24.txt           # Qwen3-235B: M1 bottom-24 at k=8
    ├── uniform_keep_24.txt         # Qwen3-235B: layer%4==0 at k=8
    └── all_k4.txt                  # Qwen3-235B: all layers at k=4
```

## Run Commands

```bash
PPL=llama.cpp/build/bin/llama-perplexity

# Qwen3-30B (single GPU)
$PPL -m qwen3/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -ngl 99 --override-kv expert_schedule=str:$SCHED

# Qwen3.5-35B (single GPU)
$PPL -m qwen35/Qwen3.5-35B-A3B-Q8_0.gguf \
  -f wikitext-2-raw/wiki.test.raw -ngl 99 --override-kv expert_schedule=str:$SCHED

# Qwen3.5-122B (2× A100-80GB, split GGUF)
CUDA_VISIBLE_DEVICES=0,1 $PPL \
  -m qwen35/122B/Qwen3.5-122B-A10B-Q8_0-00001-of-00004.gguf \
  -f wikitext-2-raw/wiki.test.raw -ngl 99 --override-kv expert_schedule=str:$SCHED

# Qwen3-235B (4× A100-80GB, split GGUF)
$PPL -m qwen3/235B/Qwen3-235B-A22B-Instruct-2507-Q8_0-00001-of-00006.gguf \
  -f wikitext-2-raw/wiki.test.raw -ngl 99 --override-kv expert_schedule=str:$SCHED
```
