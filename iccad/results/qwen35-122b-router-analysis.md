# Qwen3.5-122B-A10B Router Analysis

**Date**: 2026-03-25
**Model**: Qwen3.5-122B-A10B (Q8_0)
**Calibration**: ubergarm-imatrix-calibration-corpus-v02.txt (~424,448 tokens)
**Comparison**: Qwen3.5-35B-A3B (Q8_0)

---

## Architecture Comparison

| Parameter | Qwen3.5-35B | Qwen3.5-122B |
|-----------|-------------|--------------|
| MoE layers | 40 | **48** |
| n_expert (routed) | 256 | **256** |
| n_expert_used | 8 | **8** |
| n_embd | 2048 | **3072** |
| n_ff (per routed expert) | 512 | (TBD from tensor sizes) |
| Shared experts | Yes | **Yes** |
| Graph model file | `qwen35moe.cpp` | `qwen35moe.cpp` |
| Total params | ~35B | ~122B |
| Active params | ~3B | ~10B |

Same architecture family (shared + routed experts with sigmoid gating), but deeper (48 vs 40 layers) and wider (n_embd 3072 vs 2048).

**Note**: Layers 38 and 42 have zero-count expert indices (expert 69 and 25 respectively), indicating these experts were never activated during calibration. This is normal for 256-expert models with limited calibration data.

---

## Per-Layer Router Metrics

| Layer | SM_k4 | SM_k6 | WC_k4 | WC_k6 | MaxProb | RoutEnt |
|-------|-------|-------|-------|-------|---------|---------|
| 0 | 0.000970 | 0.000332 | 0.646 | 0.830 | 0.026 | 7.626 |
| 1 | 0.001960 | 0.000889 | 0.673 | 0.848 | 0.043 | 7.518 |
| 2 | 0.001834 | 0.000936 | 0.643 | 0.834 | 0.038 | 7.598 |
| 3 | 0.001406 | 0.000699 | 0.642 | 0.832 | 0.033 | 7.679 |
| 4 | 0.001448 | 0.000731 | 0.636 | 0.828 | 0.036 | 7.651 |
| 5 | 0.001608 | 0.000805 | 0.643 | 0.832 | 0.041 | 7.593 |
| 6 | 0.001780 | 0.000916 | 0.643 | 0.833 | 0.043 | 7.562 |
| 7 | 0.002031 | 0.001006 | 0.661 | 0.842 | 0.050 | 7.504 |
| 8 | 0.001802 | 0.000905 | 0.657 | 0.839 | 0.048 | 7.534 |
| 9 | 0.002136 | 0.001059 | 0.678 | 0.850 | 0.057 | 7.444 |
| 10 | 0.002238 | 0.001137 | 0.665 | 0.845 | 0.056 | 7.400 |
| 11 | 0.002396 | 0.001190 | 0.669 | 0.847 | 0.057 | 7.391 |
| 12 | 0.002257 | 0.001131 | 0.669 | 0.846 | 0.060 | 7.399 |
| 13 | 0.002510 | 0.001242 | 0.666 | 0.846 | 0.057 | 7.393 |
| 14 | 0.002224 | 0.001113 | 0.658 | 0.841 | 0.054 | 7.444 |
| 15 | 0.002554 | 0.001239 | 0.668 | 0.847 | 0.056 | 7.420 |
| 16 | 0.002635 | 0.001321 | 0.669 | 0.848 | 0.059 | 7.393 |
| 17 | 0.002868 | 0.001402 | 0.673 | 0.850 | 0.061 | 7.363 |
| 18 | 0.002755 | 0.001363 | 0.674 | 0.850 | 0.063 | 7.364 |
| 19 | 0.002819 | 0.001436 | 0.668 | 0.848 | 0.060 | 7.341 |
| 20 | 0.002789 | 0.001419 | 0.673 | 0.850 | 0.061 | 7.364 |
| 21 | 0.002763 | 0.001447 | 0.663 | 0.846 | 0.055 | 7.381 |
| 22 | 0.002715 | 0.001414 | 0.666 | 0.848 | 0.053 | 7.410 |
| 23 | 0.003132 | 0.001575 | 0.682 | 0.856 | 0.063 | 7.323 |
| 24 | 0.003025 | 0.001569 | 0.672 | 0.851 | 0.057 | 7.356 |
| 25 | 0.002967 | 0.001536 | 0.669 | 0.849 | 0.057 | 7.334 |
| 26 | 0.003093 | 0.001619 | 0.670 | 0.851 | 0.057 | 7.327 |
| 27 | 0.003206 | 0.001729 | 0.669 | 0.851 | 0.060 | 7.277 |
| 28 | 0.003414 | 0.001749 | 0.675 | 0.854 | 0.059 | 7.301 |
| 29 | 0.003535 | 0.001864 | 0.675 | 0.854 | 0.060 | 7.273 |
| 30 | 0.003640 | 0.001952 | 0.669 | 0.852 | 0.059 | 7.227 |
| 31 | 0.003983 | 0.002205 | 0.668 | 0.852 | 0.064 | 7.134 |
| 32 | 0.004258 | 0.002389 | 0.671 | 0.854 | 0.068 | 7.067 |
| 33 | 0.003719 | 0.002086 | 0.659 | 0.847 | 0.060 | 7.149 |
| 34 | 0.003339 | 0.001826 | 0.660 | 0.846 | 0.058 | 7.224 |
| 35 | 0.003239 | 0.001788 | 0.654 | 0.843 | 0.056 | 7.224 |
| 36 | 0.003129 | 0.001778 | 0.650 | 0.841 | 0.056 | 7.216 |
| 37 | 0.003185 | 0.001845 | 0.647 | 0.840 | 0.055 | 7.192 |
| 38 | 0.003125 | 0.001763 | 0.649 | 0.841 | 0.052 | 7.238 |
| 39 | 0.002792 | 0.001625 | 0.639 | 0.835 | 0.050 | 7.261 |
| 40 | 0.003296 | 0.001938 | 0.646 | 0.840 | 0.052 | 7.159 |
| 41 | 0.004452 | 0.002815 | 0.656 | 0.848 | 0.063 | 6.958 |
| 42 | 0.003038 | 0.001744 | 0.648 | 0.841 | 0.046 | 7.322 |
| 43 | 0.002699 | 0.001552 | 0.644 | 0.838 | 0.043 | 7.401 |
| 44 | 0.002661 | 0.001498 | 0.644 | 0.839 | 0.040 | 7.469 |
| 45 | 0.002998 | 0.001940 | 0.647 | 0.843 | 0.043 | 7.390 |
| 46 | 0.002017 | 0.001312 | 0.626 | 0.830 | 0.031 | 7.567 |
| 47 | 0.002158 | 0.001997 | 0.623 | 0.832 | 0.036 | 7.467 |

---

## Key Finding: U-Shaped Sensitivity Profile

Unlike the 35B model's monotonic depth gradient, the 122B model exhibits a **U-shaped** sensitivity pattern: both shallow AND deep layers are sensitive, while middle layers are safest.

### Layer group averages

| Segment | SM_k4 | WC_k4 | MaxProb | RoutEnt | Interpretation |
|---------|-------|-------|---------|---------|----------------|
| 0-11 (early) | 0.00180 | 0.655 | 0.044 | 7.542 | **Low SM** — routing uncertain at boundary |
| 12-23 (mid-early) | 0.00267 | 0.669 | 0.059 | 7.383 | Moderate — routing more confident |
| 24-35 (mid-late) | 0.00345 | 0.668 | 0.060 | 7.241 | **Highest SM** — safest for pruning |
| 36-47 (deep) | 0.00296 | 0.643 | 0.047 | 7.303 | SM moderate, but **lowest WC** |

The U-shape arises because SM and WC disagree on which end is sensitive:
- **SM says**: early layers are most sensitive (lowest score margins → uncertain routing)
- **WC says**: deep layers are most sensitive (lowest weight concentration → top-4 captures least weight)
- **Combined**: both extremes are flagged, middle layers are consistently safest

### SM-WC decorrelation

| Model | SM_k4 ↔ WC_k4 | SM_k4 ↔ MaxProb | SM_k4 ↔ RoutEnt |
|-------|----------------|------------------|-------------------|
| Qwen3.5-35B | +0.76 | +0.78 | -0.85 |
| **Qwen3.5-122B** | **+0.37** | **+0.67** | **-0.96** |

In the 122B model, SM and WC are only weakly correlated (ρ = +0.37 vs 0.76 in 35B). This means score margin (routing confidence at the boundary) and weight concentration (how peaked the top-k distribution is) capture different aspects of sensitivity at this scale.

---

## Per-Layer Sensitivity Ranking

Combined score = average rank across SM_k4, SM_k6, WC_k4, WC_k6, MaxProb, RoutEntropy. Lower combined rank = more sensitive.

### Top 12 most sensitive (should keep k=8)

| Rank | Layer | SM_k4 | WC_k4 | MaxProb | Key reason |
|------|-------|-------|-------|---------|------------|
| 1 | 3 | 0.001406 | 0.642 | 0.033 | Very low SM + WC |
| 2 | 4 | 0.001448 | 0.636 | 0.036 | Very low SM + lowest WC in early group |
| 3 | 0 | 0.000970 | 0.646 | 0.026 | **Lowest SM of all** — extreme outlier |
| 4 | 5 | 0.001608 | 0.643 | 0.041 | Low SM |
| 5 | 46 | 0.002017 | 0.626 | 0.031 | **Very low WC** + low MaxProb |
| 6 | 6 | 0.001780 | 0.643 | 0.043 | Low SM |
| 7 | 2 | 0.001834 | 0.643 | 0.038 | Low SM |
| 8 | 8 | 0.001802 | 0.657 | 0.048 | Low SM |
| 9 | 47 | 0.002158 | 0.623 | 0.036 | **Lowest WC of all** |
| 10 | 44 | 0.002661 | 0.644 | 0.040 | Low WC + low MaxProb |
| 11 | 7 | 0.002031 | 0.661 | 0.050 | Low SM |
| 12 | 43 | 0.002699 | 0.644 | 0.043 | Low WC |

**Both shallow (0-8) and deep (43-47) layers are selected.** This is qualitatively different from the 35B model, where only deep layers were sensitive.

### Per-metric top 12

| Metric | Top 12 most sensitive |
|--------|----------------------|
| SM_k4 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 46, 47 |
| SM_k6 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14 |
| WC_k4 | 0, 2, 3, 4, 5, 6, 39, 40, 43, 44, 46, 47 |
| MaxProb | 0, 1, 2, 3, 4, 5, 6, 43, 44, 45, 46, 47 |
| RoutEnt | 0, 1, 2, 3, 4, 5, 6, 7, 8, 44, 46, 47 |

SM flags early layers; WC flags deep layers; MaxProb and RoutEnt flag both extremes.

---

## Recommended Schedules

### Three-tier schedule (k=8 / k=6 / k=4)

| Tier | top_k | Layers | Count |
|------|-------|--------|-------|
| 1 | 8 | 0, 2, 3, 4, 5, 6, 7, 8, 43, 44, 46, 47 | 12 |
| 2 | 6 | 1, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 36, 37, 38, 39, 40, 42, 45 | 18 |
| 3 | 4 | 17, 18, 19, 20, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 41 | 18 |

**Computation saving**: 12 layers at full cost + 18 at 75% + 18 at 50% = **31.25% MoE expert FLOP reduction**.

### Two-tier schedule (k=8 vs k=4)

| Tier | top_k | Layers | Count |
|------|-------|--------|-------|
| 1 | 8 | 0, 2, 3, 4, 5, 6, 7, 8, 43, 44, 46, 47 | 12 |
| 2 | 4 | all others | 36 |

**Computation saving**: 12 at full + 36 at 50% = **37.5% MoE expert FLOP reduction**.

---

## Notable Anomalies

### Layer 0 — extreme SM outlier

SM_k4 = 0.000970, the lowest of all 48 layers by a large margin (next lowest: layer 3 at 0.001406). SM_k6 = 0.000332 is even more extreme. The first MoE layer routes with very low confidence — possibly because it receives the least-processed input representation.

### Layer 41 — safest layer

SM_k4 = 0.004452, the highest of all layers. Layer 41 routes with very high confidence and has the lowest routing entropy (6.958). This is the only layer with RoutEnt < 7.0.

### Layers 46-47 — WC outliers

WC_k4 values of 0.626 and 0.623 are the lowest of all layers, meaning top-4 experts capture only ~62% of the top-8 weight. However, their SM_k4 is moderate (0.002017, 0.002158), suggesting the boundary gap is reasonable but the overall weight distribution is unusually flat.

Layer 47 has SM_k6/SM_k4 = 0.925 — the gap at the k=6 boundary is almost as large as at k=4. This is similar to Qwen3.5-35B layer 39's pattern.

### Layer 41 vs deep layers

Layer 41 is by far the safest layer despite being deep. This breaks the U-shape pattern locally — it's an isolated peak of routing confidence surrounded by sensitive deep layers.

---

## Cross-Scale Comparison: 35B vs 122B

### Sensitivity pattern reversal

| Aspect | Qwen3.5-35B (40 layers) | Qwen3.5-122B (48 layers) |
|--------|------------------------|--------------------------|
| SM range | 0.0014 – 0.0033 | 0.0010 – 0.0045 |
| SM depth trend | **Monotonic decrease** (deeper = lower SM) | **U-shaped** (early + deep both low) |
| Most sensitive (SM) | Layers 35-39 (deep) | Layers 0-8 (early) |
| Most sensitive (WC) | Layers 35-39 (deep) | Layers 39, 46-47 (deep) |
| SM ↔ WC correlation | +0.76 (strong) | **+0.37** (weak) |
| Safest layers | Layers 2, 10, 11 (early) | Layers 29-32 (mid-late) |
| Metric agreement | All metrics agree | **SM and WC disagree on depth direction** |

### Why the pattern differs

The 35B model had all metrics agreeing because every metric decreased monotonically with depth. In the 122B model, SM (routing confidence) is lowest in early layers while WC (weight peakedness) is lowest in deep layers. This creates a situation where:

- **Early layers**: the router is uncertain about which experts to pick (low SM), but once it picks, the top-4 do capture a reasonable share (moderate WC)
- **Deep layers**: the router is relatively confident about its top choice (higher SM), but the overall distribution is flat — many experts get similar weight (low WC)

These are qualitatively different forms of sensitivity. The combined ranking handles this by flagging both types.

### Implication for methodology

The depth-reversal between 35B and 122B confirms that **sensitivity patterns are model-specific, not architecture-specific**. Even within the same architecture family (Qwen3.5), scaling up changes which layers are most sensitive. This validates the approach of running per-model router analysis rather than assuming patterns transfer across scales.

---

## Next Steps

1. **Qwen3-235B analysis**: Compare with this model to test whether the U-shape pattern is a scale effect (large models in general) or architecture-specific (Qwen3.5 only).
2. **Perplexity validation**: Run perplexity with the three-tier schedule to validate whether the router-metric-based ranking correctly identifies layers where pruning hurts quality.
3. **SM vs WC disagreement layers**: Layers where SM says safe but WC says sensitive (or vice versa) are the most uncertain — prioritize these for perplexity validation.
