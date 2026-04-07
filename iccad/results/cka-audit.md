# CKA Methodology Audit

Systematic review of the CKA (Centered Kernel Alignment) pipeline used for per-layer
importance ranking in the ICCAD project.

## 1. Implementation Verification

The `linear_CKA` function in `compute_cka.py` and the streaming equivalent in
`imatrix-hsic.cpp` both implement:

```
CKA(X, Y) = ||Y_c^T X_c||_F^2 / sqrt(||X_c^T X_c||_F^2 * ||Y_c^T Y_c||_F^2)
```

where X_c = X - mean(X), Y_c = Y - mean(Y).

This is equivalent to `HSIC(K, L) / sqrt(HSIC(K,K) * HSIC(L,L))` with linear kernel,
which is the standard CKA definition from Kornblith et al. (2019). **Verified correct.**

When n_samples < n_features, `compute_cka.py` uses the kernel-space formulation:
`K = X_c @ X_c.T` (n x n) instead of `X_c.T @ X_c` (p x p), giving identical results
with ~4.5x fewer FLOPs.

## 2. Bootstrap Ranking Stability (200 resamples, 80% chunks)

Tests whether the M1 top-12/top-10 selection is robust to data variation.

### Qwen3-30B (48 layers, 840 chunks)

| Layer | M1 mean | M1 std | Med rank | Rank std | Top-12 freq |
|-------|---------|--------|----------|----------|-------------|
| 46    | 0.2300  | 0.0437 | 1.0      | 0.79     | 100.0%      |
| 45    | 0.1942  | 0.0356 | 2.0      | 1.14     | 100.0%      |
| 44    | 0.1522  | 0.0389 | 3.0      | 2.59     | 97.5%       |
| 37    | 0.1480  | 0.0350 | 4.0      | 1.75     | 99.5%       |
| 42    | 0.1377  | 0.0196 | 5.0      | 0.61     | 100.0%      |
| 43    | 0.1331  | 0.0304 | 6.0      | 2.22     | 99.0%       |
| 12    | 0.1026  | 0.0122 | 8.0      | 1.49     | 100.0%      |
| 39    | 0.1010  | 0.0216 | 8.0      | 3.58     | 75.5%       |
| 10    | 0.0962  | 0.0109 | 10.0     | 3.87     | 68.0%       |
| 8     | 0.0941  | 0.0107 | 11.0     | 1.84     | 68.5%       |
| 11    | 0.0940  | 0.0131 | 12.0     | 3.43     | 60.0%       |
| 3     | 0.0923  | 0.0161 | 13.5     | 5.28     | 40.5%       |
| 6     | 0.0910  | 0.0109 | 13.0     | 5.74     | 47.5%       |
| 40    | 0.0920  | 0.0228 | 14.0     | 6.22     | 46.0%       |
| 20    | 0.0916  | 0.0122 | 13.0     | 2.47     | 47.0%       |

- Original top-12: [46, 45, 37, 44, 42, 43, 12, 10, 39, 8, 6, 3]
- Exact match frequency: **5.0%** (low — borderline layers swap often)
- Mean overlap with original: **10.0/12** (top-7 are very stable, positions 8-12 volatile)
- Core stable set (>95% freq): {46, 45, 44, 37, 42, 43, 12} — 7 layers
- Borderline zone (40-70% freq): {3, 6, 8, 10, 11, 20, 39, 40} — 8 candidates for 5 slots

### Qwen3.5-35B (40 layers, 840 chunks)

| Layer | M1 mean | M1 std | Med rank | Rank std | Top-12 freq |
|-------|---------|--------|----------|----------|-------------|
| 38    | 0.2860  | 0.0450 | 1.0      | 0.21     | 100.0%      |
| 37    | 0.1644  | 0.0328 | 3.0      | 2.80     | 98.0%       |
| 35    | 0.1562  | 0.0396 | 4.0      | 5.52     | 84.0%       |
| 36    | 0.1519  | 0.0295 | 5.0      | 1.06     | 100.0%      |
| 34    | 0.1495  | 0.0495 | 5.0      | 11.92    | 75.5%       |
| 19    | 0.1271  | 0.0224 | 6.0      | 5.70     | 84.0%       |
| 26    | 0.1148  | 0.0272 | 8.0      | 1.66     | 99.0%       |
| 9     | 0.1184  | 0.0386 | 9.0      | 3.03     | 90.0%       |
| 25    | 0.1071  | 0.0268 | 11.0     | 1.75     | 79.5%       |
| 23    | 0.1085  | 0.0235 | 11.0     | 7.50     | 58.0%       |

- Original top-12: [38, 37, 35, 34, 36, 19, 26, 23, 9, 25, 21, 33]
- Exact match frequency: **11.5%**
- Mean overlap with original: **9.5/12**
- Core stable set (>95% freq): {38, 37, 36, 26} — 4 layers
- Note: layer 34 has very high rank std (11.92) despite 75.5% top-12 freq — bimodal

### Bootstrap Summary

The top-5-7 layers are robust across both models. Positions 8-12 are volatile, with
multiple candidate layers having similar M1 values. This is expected: the M1 distribution
has a long flat middle, so small perturbations swap layers in/out.

**Practical impact**: The core layers that should always keep k=8 are clear and stable.
The borderline layers (positions 8-12) have ~40-70% probability of being in the top set,
and swapping them has minimal impact because their M1 values are very close.

## 3. Token-Level vs Chunk-Averaged CKA

### Background

The original pipeline averages 512 token hidden states per chunk into 1 mean vector,
then computes CKA on ~840 chunk means (840 samples x 2048 features). This loses
intra-chunk token-level variation.

The streaming CKA mode accumulates covariance matrices token-by-token and computes
exact CKA over all ~430K individual tokens. Implementation uses float32 intermediate
with cache-tiled matrix multiplication + OpenMP parallelization across layers. Memory:
~2.5 GB for accumulated covariance matrices + ~200 MB per-chunk buffers.

### Results

#### Qwen3-30B (430,080 tokens vs 840 chunk means)

| Metric | Value |
|--------|-------|
| Spearman rho (M1 ranking) | **0.595** |
| Top-12 overlap | **7/12 (58%)** |

```
Chunk-avg top-12: [46, 45, 37, 44, 42, 43, 12, 10, 39, 8,  6,  3]
Token-lvl top-12: [45, 44, 42, 43, 37, 40, 39,  8, 34, 38, 41,  4]
Common (7):       {8, 37, 39, 42, 43, 44, 45}
Only chunk:       {3, 6, 10, 12, 46}
Only token:       {4, 34, 38, 40, 41}
```

**Key difference**: Layer 46 drops from rank 1 (chunk-averaged) to ~rank 28 (token-level).
Its token-level S_in is very low (0.020 vs 0.435 chunk-averaged), meaning individual
token representations at layer 46 are very dissimilar from the input embedding — but
chunk *means* at layer 46 still correlate with input chunk means.

This suggests layer 46's chunk-averaged importance reflects inter-document variation
(different topics produce different mean vectors) rather than per-token processing.

#### Qwen3.5-35B (424,448 tokens vs 840 chunk means)

| Metric | Value |
|--------|-------|
| Spearman rho (M1 ranking) | **0.641** |
| Top-12 overlap | **8/12 (67%)** |

```
Chunk-avg top-12: [38, 37, 35, 34, 36, 19, 26, 23,  9, 25, 21, 33]
Token-lvl top-12: [38, 37, 36, 34, 35, 32, 33, 31, 19, 29,  9, 28]
Common (8):       {9, 19, 33, 34, 35, 36, 37, 38}
Only chunk:       {21, 23, 25, 26}
Only token:       {28, 29, 31, 32}
```

Better agreement than Qwen3. Top-5 layers (38, 37, 36, 35, 34) are consistent across
both methods. Differences are in the borderline layers (positions 6-12).

### CKA Value Magnitudes

Token-level CKA values are ~3-5x smaller than chunk-averaged values:

| Layer | Chunk S_in | Token S_in | Chunk S_out | Token S_out |
|-------|-----------|-----------|------------|------------|
| Qwen3 L45 | 0.52 | 0.11 | 0.49 | 0.25 |
| Qwen3 L46 | 0.43 | 0.02 | 0.75 | 0.04 |
| Qwen3.5 L38 | 0.37 | 0.10 | 0.64 | 0.29 |

This is expected: averaging tokens removes intra-chunk variance (noise from individual
token positions), inflating inter-chunk signal and thus CKA. The token-level CKA
captures the full variance landscape.

## 4. Conclusions

1. **CKA implementation is correct.** Both the Python and C++ implementations match
   the standard CKA formula.

2. **Bootstrap stability is moderate.** The top-7 layers are very stable (>95% frequency),
   but positions 8-12 are volatile. For practical use, this means the core set of
   important layers is reliable, and borderline layers have similar importance anyway.

3. **Per-chunk averaging changes rankings.** Spearman rho between token-level and
   chunk-averaged M1 is ~0.6 (moderate). The top-5 layers are consistent, but specific
   layer selections differ (58-67% top-12 overlap).

4. **Layer 46 (Qwen3) is a chunk-averaging artifact.** It appears as the most important
   layer under chunk-averaged CKA but drops to ~rank 28 under token-level CKA. Its
   chunk-level importance reflects inter-document variation, not per-token processing.
   This is the single largest discrepancy.

5. **Practical impact on PPL validation.** Our PPL experiments used the chunk-averaged
   M1 top-12, which included layer 46 and excluded layers like 40, 34, 38, 41. The
   token-level ranking suggests a different schedule might perform better. However,
   the chunk-averaged schedule still outperformed uniform and bottom-12, so CKA-guided
   selection is directionally correct even with the averaging approximation.

6. **Recommendation.** For future experiments, prefer token-level CKA (streaming mode)
   as the primary ranking. Use chunk-averaged CKA as a fast approximation for initial
   screening. The streaming CKA tool (`--streaming-cka` flag) runs in ~75 minutes per
   model with the OpenMP-optimized implementation.

## Files

| File | Description |
|------|-------------|
| `tools/imatrix-hsic/imatrix-hsic.cpp` | StreamingCKA mode (`--streaming-cka` flag) |
| `tools/imatrix-hsic/CMakeLists.txt` | Added OpenMP linkage |
| `iccad/scripts/compute_cka.py` | Bootstrap analysis (`--bootstrap N`), kernel-space CKA optimization |
| `qwen3/cka-bootstrap.txt` | Qwen3 bootstrap results (200 resamples) |
| `qwen35/cka-bootstrap.txt` | Qwen3.5 bootstrap results (200 resamples) |
