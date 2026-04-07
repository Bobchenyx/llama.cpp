# Per-Layer top_k Perplexity Validation — Qwen3.5-35B-A3B

## Setup

- Model: Qwen3.5-35B-A3B-Q8_0.gguf (40 MoE layers, default k=8, 256 experts + shared experts)
- Benchmark: wikitext-2-raw (wiki.test.raw), ctx=512, 584 chunks
- Mechanism: per-layer `n_expert_used` schedule via `--override-kv expert_schedule=str:<path>`
- All schedule runs: 29 layers at k=4, 11 layers at k=8 (10 selected + layer 39 always kept)
- Layer 39 excluded from all schedules (last layer before output, always k=8)
- Architecture: hybrid model (30 recurrent + 10 full attention layers), all layers have MoE FFN

## Results

| Run | Setting | 10 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 40 | 6.5359 ± 0.042 | — |
| A2  | **token-level CKA M1 top-10** | [19,29,31,32,33,34,35,36,37,38] | **7.0799 ± 0.046** | +0.544 |
| A1  | chunk-averaged CKA M1 top-10 | [9,19,23,25,26,34,35,36,37,38] | 7.1060 ± 0.046 | +0.570 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36] | 7.2591 ± 0.047 | +0.723 |
| B   | CKA M1 bottom-10 | [0,4,5,6,8,10,12,15,16,20] | 7.3529 ± 0.049 | +0.817 |
| B2  | all k=4 | none | 7.5784 ± 0.050 | +1.043 |

## Ordering

```
B1 (all k=8)  <  A2 (token M1)  <  A1 (chunk M1)  <  C (uniform)  <  B (bot-10)  <  B2 (all k=4)
   6.54            7.08              7.11               7.26            7.35            7.58
```

## Analysis

1. **Token-level CKA is best**: Token-level M1 top-10 (A2) outperforms chunk-averaged M1 (A1)
   by 0.026 PPL points (7.08 vs 7.11). The improvement is smaller than Qwen3 (0.05) because
   Qwen3.5's token-level and chunk-averaged rankings have higher agreement (ρ=0.64 vs 0.60).

2. **CKA ranking validated on Qwen3.5**: Both CKA-guided selections beat uniform (7.26)
   and bottom-10 (7.35). Token-level PPL degradation is 52% of the all-k4 gap (0.54 vs 1.04).

3. **PPL recovery**: Token-level CKA-guided selection recovers 48% of the PPL gap
   (1.043 total, 0.499 recovered = 1.043 - 0.544).

4. **Layer importance pattern**: Token-level top-10 layers are heavily concentrated in the
   deep end (29, 31-38), with only layer 19 as a mid-range exception. Chunk-averaged top-10
   had more mid-range layers (9, 23, 25, 26) that token-level CKA correctly deprioritizes.

5. **Shared experts buffer the impact**: Qwen3.5's shared experts (which are NOT affected by
   top_k reduction) provide a floor for each layer's FFN output. This means the total PPL gap
   between all-k8 and all-k4 is smaller (1.04 for Qwen3.5 vs 1.26 for Qwen3).

## Cross-Model Comparison

| Metric | Qwen3-30B (48 layers) | Qwen3.5-35B (40 layers) |
|--------|----------------------|------------------------|
| Baseline PPL (all k=8) | 7.3598 | 6.5359 |
| All k=4 PPL | 8.6199 | 7.5784 |
| PPL gap (all-k4 - all-k8) | 1.2601 | 1.0425 |
| Token-level CKA top-N PPL | **7.9370** (N=12) | **7.0799** (N=10) |
| Chunk-averaged CKA top-N PPL | 7.9905 (N=12) | 7.1060 (N=10) |
| Token vs chunk ΔPPL | -0.054 | -0.026 |
| Token-level recovery % | 54.0% | 47.8% |
| Bottom-N PPL | 8.2323 | 7.3529 |
| Uniform PPL | 8.1357 | 7.2591 |
| Keep ratio | 12/48 = 25% | 10/40 = 25% |
| Ordering correct | Yes | Yes |

Both models show the same ordering: B1 < token-CKA < chunk-CKA < uniform < bottom < all-k4.
Token-level CKA consistently outperforms chunk-averaged CKA, but both are valid options —
the difference is small (0.03-0.05 PPL). CKA M1 generalizes across architectures
(homogeneous vs hybrid, 128 vs 256 experts, with and without shared experts).

## Code Change

Same 1-line change as Qwen3 — `src/models/qwen35moe.cpp:379`:
```cpp
n_expert, cparams.warmup ? n_expert_used : (int64_t)hparams.n_expert_used_arr[il],
```

## Run Commands

```bash
MODEL=qwen35/Qwen3.5-35B-A3B-Q8_0.gguf
DATA=wikitext-2-raw/wiki.test.raw
PPL=llama.cpp/build/bin/llama-perplexity
SCHED=llama.cpp/iccad/schedules/qwen35

$PPL -m $MODEL -f $DATA -ngl 38                                                          # B1
$PPL -m $MODEL -f $DATA -ngl 38 --override-kv qwen35moe.expert_used_count=int:4           # B2
$PPL -m $MODEL -f $DATA -ngl 38 --override-kv expert_schedule=str:$SCHED/token_m1_keep_top10.txt # A2
$PPL -m $MODEL -f $DATA -ngl 38 --override-kv expert_schedule=str:$SCHED/m1_keep_top10.txt       # A1
$PPL -m $MODEL -f $DATA -ngl 38 --override-kv expert_schedule=str:$SCHED/m1_keep_bot10.txt       # B
$PPL -m $MODEL -f $DATA -ngl 38 --override-kv expert_schedule=str:$SCHED/uniform_keep_10.txt     # C
```
