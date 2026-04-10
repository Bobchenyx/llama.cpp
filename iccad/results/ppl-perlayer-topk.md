# Per-Layer top_k Perplexity Validation

## Setup

- Benchmark: wikitext-2-raw (wiki.test.raw), ctx=512, 584 chunks
- Mechanism: per-layer `n_expert_used` schedule via `--override-kv expert_schedule=str:<path>`
- Schedule format: `layer_id top_k` per line; unlisted layers keep model default (k=8)
- Last MoE layer always kept at k=8 (layer 47 for Qwen3, layer 39 for Qwen3.5)

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

## Analysis

**Ordering** (consistent across both models):
```
all-k8  <  token-CKA  <  chunk-CKA  <  uniform  <  bottom-N  <  all-k4
```

1. **Token-level CKA is best**: Token-level M1 outperforms chunk-averaged by 0.03-0.05 PPL.
   Both methods are valid; token-level is recommended when compute allows (~75 min vs ~15 min).

2. **CKA ranking validated**: CKA M1 top-N beats uniform spacing by 0.15-0.20 PPL and
   bottom-N by 0.28-0.87 PPL. Layer importance is NOT uniform across depth.

3. **M1 ≈ M2**: Two CKA metrics give identical PPL (Δ=0.0007). M1 preferred (ratio stability).

4. **PPL recovery**: Token-level CKA-guided selection recovers 48-54% of the PPL gap
   while pruning 73-75% of layers to k=4. Only 7-8% PPL increase from baseline.

5. **Shared experts buffer Qwen3.5**: Total PPL gap is smaller (1.04 vs 1.26) despite
   more experts and lower router confidence.

---

## Cross-Model Comparison

| Metric | Qwen3-30B (48 layers) | Qwen3.5-35B (40 layers) |
|--------|----------------------|------------------------|
| Baseline PPL (all k=8) | 7.3598 | 6.5359 |
| All k=4 PPL | 8.6199 | 7.5784 |
| PPL gap | 1.2601 | 1.0425 |
| Token-level CKA PPL | **7.9370** | **7.0799** |
| Chunk-averaged CKA PPL | 7.9905 | 7.1060 |
| Token vs chunk ΔPPL | -0.054 | -0.026 |
| Token-level recovery % | 54.0% | 47.8% |
| Keep ratio | 12/48 = 25% | 10/40 = 25% |

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
├── token_m1_keep_top12.txt         # Qwen3: token-level CKA M1 top-12 at k=8
├── m1_keep_top12.txt               # Qwen3: chunk-averaged M1 top-12 at k=8
├── m2_keep_top12.txt               # Qwen3: chunk-averaged M2 top-12 at k=8
├── m1_keep_bot12.txt               # Qwen3: M1 bottom-12 at k=8
├── uniform_keep_12.txt             # Qwen3: layer%4==0 at k=8
└── qwen35/
    ├── token_m1_keep_top10.txt     # Qwen3.5: token-level M1 top-10 at k=8
    ├── m1_keep_top10.txt           # Qwen3.5: chunk-averaged M1 top-10 at k=8
    ├── m1_keep_bot10.txt           # Qwen3.5: M1 bottom-10 at k=8
    └── uniform_keep_10.txt         # Qwen3.5: layer%4==0 at k=8

## Run Commands

```bash
# Qwen3
MODEL=qwen3/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
PPL=llama.cpp/build/bin/llama-perplexity
$PPL -m $MODEL -f wikitext-2-raw/wiki.test.raw -ngl 99 --override-kv expert_schedule=str:$SCHED

# Qwen3.5
MODEL=qwen35/Qwen3.5-35B-A3B-Q8_0.gguf
$PPL -m $MODEL -f wikitext-2-raw/wiki.test.raw -ngl 38 --override-kv expert_schedule=str:$SCHED
```
