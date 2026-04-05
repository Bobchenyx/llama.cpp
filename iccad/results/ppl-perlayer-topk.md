# Per-Layer top_k Perplexity Validation

## Setup

- Model: Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf (48 MoE layers, default k=8)
- Benchmark: wikitext-2-raw (wiki.test.raw), ctx=512, 584 chunks
- Mechanism: per-layer `n_expert_used` schedule via `--override-kv expert_schedule=str:<path>`
- All schedule runs: 35 layers at k=4, 13 layers at k=8 (12 selected + layer 47 always kept)
- Layer 47 excluded from all schedules (last layer before output, always k=8)

## Results

| Run | Setting | 12 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 48 | 7.3598 ± 0.052 | — |
| A1  | CKA M1 top-12 | [3,6,8,10,12,37,39,42,43,44,45,46] | 7.9905 ± 0.057 | +0.631 |
| A2  | CKA M2 top-12 | [0,8,10,11,12,37,39,42,43,44,45,46] | 7.9912 ± 0.057 | +0.631 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36,40,44] | 8.1357 ± 0.058 | +0.776 |
| B   | CKA M1 bottom-12 | [0,5,9,19,21,26,28,31,32,33,34,38] | 8.2323 ± 0.059 | +0.873 |
| B2  | all k=4 | none | 8.6199 ± 0.062 | +1.260 |

## Ordering

```
B1 (all k=8)  <  A1 (M1 top-12)  ≈  A2 (M2 top-12)  <  C (uniform)  <  B (M1 bot-12)  <  B2 (all k=4)
   7.36            7.99                7.99                8.14            8.23               8.62
```

## Analysis

1. **CKA ranking validated**: M1 and M2 both identify better layers than uniform spacing or
   random (bottom-12) selection. PPL degradation is 50% of all-k4 (0.63 vs 1.26).

2. **M1 ≈ M2**: The two metrics produce nearly identical PPL (Δ=0.0007). This is expected given
   their high correlation (Spearman ρ=0.886). M1 is recommended as the primary metric since it
   avoids the ratio instability of M2 when S_in is small.

3. **Uniform baseline**: Uniform spacing (layer%4==0) is worse than CKA-guided selection by
   0.15 PPL points, showing that layer importance is NOT uniform across depth.

4. **Bottom-12 worst case**: Keeping the CKA-identified least important layers at k=8 gives
   only marginal improvement over all-k4 (8.23 vs 8.62), confirming these layers are indeed
   less sensitive to expert reduction.

5. **PPL recovery**: CKA-guided 12/48 selection recovers 50% of the PPL gap
   (1.26 total, 0.63 recovered). With 75% of layers pruned to k=4, only 8.6% PPL increase.

## Implementation Notes

- Bug found and fixed in `llama-graph.cpp:1527`: `build_moe_ffn` used global `hparams.n_expert_used`
  for expert aggregation loop, causing OOB reads when per-layer k < global k. Fixed to use
  `std::min(n_expert_used, (int64_t)hparams.n_expert_used)`.
- Layer 47 must always keep default k — setting it to k=4 causes catastrophic PPL (766+).
