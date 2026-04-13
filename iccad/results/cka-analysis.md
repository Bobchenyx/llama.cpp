# CKA Layer Importance Analysis

## Method

Linear CKA (Centered Kernel Alignment, Kornblith et al. 2019) measures representational
similarity between layer hidden states. For each MoE layer i:
- S_in(i) = CKA(X_embed, H_i) — similarity to input
- S_out(i) = CKA(H_i, H_last) — similarity to output
- **M1 = S_out × (1 - S_in)** — primary importance metric

Two computation modes:
- **Chunk-averaged**: average 512 tokens per chunk → CKA on ~840 chunk means (~15 min)
- **Token-level** (streaming): accumulate covariance matrices token-by-token, exact CKA
  over ~430K tokens (~75 min). Uses `--streaming-cka` flag in `llama-imatrix-hsic`.

Implementation: `tools/imatrix-hsic/imatrix-hsic.cpp` (C++, OpenMP) and
`iccad/scripts/compute_cka.py` (Python, offline analysis + bootstrap).

---

## Bootstrap Ranking Stability (200 resamples, 80% chunks)

### Qwen3-30B (48 layers, 840 chunks)

- Core stable set (>95% top-12 freq): **{46, 45, 44, 37, 42, 43, 12}** — 7 layers
- Borderline zone (40-70% freq): {3, 6, 8, 10, 11, 20, 39, 40} — 8 candidates for 5 slots
- Mean overlap with original top-12: 10.0/12
- Exact match frequency: 5.0%

### Qwen3.5-35B (40 layers, 840 chunks)

- Core stable set (>95% freq): **{38, 37, 36, 26}** — 4 layers
- Borderline zone: {9, 19, 23, 25, 34, 35} — varied frequency
- Mean overlap with original top-12: 9.5/12
- Note: layer 34 has rank std = 11.92 (bimodal)

**Summary**: Top 5-7 layers are very stable. Positions 8-12 are volatile — multiple
candidates have similar M1 values, so swaps have minimal PPL impact.

---

## Token-Level vs Chunk-Averaged CKA

### Qwen3-30B (430,080 tokens vs 840 chunk means)

| Metric | Value |
|--------|-------|
| Spearman ρ (M1 ranking) | **0.595** |
| Top-12 overlap | **7/12 (58%)** |

```
Chunk-avg top-12: [46, 45, 37, 44, 42, 43, 12, 10, 39, 8,  6,  3]
Token-lvl top-12: [45, 44, 42, 43, 37, 40, 39,  8, 34, 38, 41,  4]
```

**Key discrepancy**: Layer 46 drops from rank 1 (chunk) to rank ~28 (token-level).
Its chunk importance reflects inter-document mean variation, not per-token processing —
a chunk-averaging artifact.

### Qwen3.5-35B (424,448 tokens vs 840 chunk means)

| Metric | Value |
|--------|-------|
| Spearman ρ (M1 ranking) | **0.641** |
| Top-12 overlap | **8/12 (67%)** |

```
Chunk-avg top-12: [38, 37, 35, 34, 36, 19, 26, 23,  9, 25, 21, 33]
Token-lvl top-12: [38, 37, 36, 34, 35, 32, 33, 31, 19, 29,  9, 28]
```

Better agreement — top-5 layers consistent across both methods.

### CKA Value Magnitudes

Token-level values are ~3-5x smaller than chunk-averaged (averaging removes intra-chunk
variance, inflating inter-chunk signal):

| Layer | Chunk S_out | Token S_out |
|-------|------------|------------|
| Qwen3 L45 | 0.49 | 0.25 |
| Qwen3 L46 | 0.75 | 0.04 |
| Qwen3.5 L38 | 0.64 | 0.29 |

---

## PPL Validation: Token vs Chunk CKA Rankings

| Model | Token-level PPL | Chunk-averaged PPL | ΔPPL |
|-------|----------------|-------------------|------|
| Qwen3-30B | **7.9370** | 7.9905 | -0.054 |
| Qwen3.5-35B | **7.0799** | 7.1060 | -0.026 |

Token-level is consistently better, but both are valid options.

---

## Large Model CKA Rankings (chunk-averaged)

### Qwen3.5-122B (48 layers, n_embd=3072, 829 chunks)

```
M1 top-12: [42, 43, 44, 39, 40, 46, 45, 38, 37, 35, 31, 36]
   bot-12: [16, 20, 28, 9, 4, 14, 12, 27, 3, 7, 22, 8]
```

M1 range: [0.066, 0.195]. Top layers concentrated in deep region (31-46).
S_in range: [0.13, 0.68], S_out range: [0.08, 0.48].
M1 vs M2 ρ = 0.801, M1 vs M3 ρ = 0.098 (M3 poorly correlated).

### Qwen3-235B (94 layers, n_embd=4096, 840 chunks)

```
M1 top-12: [91, 92, 85, 90, 23, 25, 7, 0, 27, 87, 89, 21]
   bot-12: [3, 2, 70, 73, 68, 67, 71, 12, 63, 69, 74, 32]
```

M1 range: [0.012, 0.247]. **Bimodal distribution**: deep layers (85-92) and early
layers (0, 7, 21-27) both in top-12. Middle layers (60-75) are consistently unimportant.
S_in range: [0.04, 0.80], S_out range: [0.01, 0.67].
M1 vs M2 ρ = 0.961, M1 vs M3 ρ = 0.580.

### Cross-Scale Consistency

| Model | Layers | Top-12 depth profile |
|-------|--------|---------------------|
| Qwen3-30B | 48 | Deep (37-46) + scattered early (3, 6, 8, 10, 12) |
| Qwen3-235B | 94 | Deep (85-92) + early (0, 7, 21-27) — bimodal |
| Qwen3.5-35B | 40 | Deep (33-38) + mid (19, 23, 25, 26) |
| Qwen3.5-122B | 48 | Deep (35-46) — concentrated in final 1/3 |

---

## Probe Point Comparison: ffn_moe_out vs l_out

Investigated whether probing full layer output (`l_out`/`post_moe`, including attention +
residual + shared expert) gives different CKA rankings vs `ffn_moe_out` (pure MoE expert
weighted output). Tested on 30B and 35B using `--probe-l-out` flag.

### Results

| | ffn_moe_out M1 top-12 | l_out M1 top-12 | Overlap |
|---|---|---|---|
| Qwen3-30B | [46,45,37,44,42,43,12,10,39,8,6,3] | [46,45,44,43,42,40,41,18,31,19,32,20] | 5/12 (42%) |
| Qwen3.5-35B | [38,37,35,34,36,19,26,23,9,25,21,33] | [26,27,24,25,22,18,23,30,29,28,21,20] | 4/12 (33%) |

### Why l_out is depth-confounded

l_out includes the full residual stream, which accumulates information from all prior layers.

| | ffn_moe_out S_in range | l_out S_in range | l_out M1 vs M3 ρ |
|---|---|---|---|
| Qwen3-30B | [0.04, 0.80] (wide) | [0.66, 0.85] (narrow) | **0.998** |
| Qwen3.5-35B | [0.02, 0.76] (wide) | [0.50, 0.86] (narrow) | **0.983** |

With l_out, S_in is near-constant (residual always carries most info), so M1 ≈ S_out,
which monotonically increases with depth. The rankings become trivially depth-ordered.

**Conclusion**: `ffn_moe_out` is the correct probe point for top_k scheduling — it
isolates the MoE contribution to information flow, producing non-trivial rankings that
identify genuinely important layers regardless of depth.

---

## Conclusions

1. **CKA implementation correct.** Python and C++ match the standard linear CKA formula.

2. **Bootstrap: top 5-7 layers robust**, positions 8-12 volatile but with similar M1 values.

3. **Chunk averaging changes rankings** (ρ ≈ 0.6, 58-67% top-12 overlap). Top-5 consistent.

4. **Layer 46 (Qwen3) is a chunk-averaging artifact** — rank 1 chunk, rank ~28 token-level.

5. **Token-level PPL is slightly better** (ΔPPL = -0.03 to -0.05). Both methods valid;
   token-level recommended when compute allows.

6. **CKA scales to large models.** 122B and 235B chunk-averaged CKA rankings validate
   at PPL level (CKA < uniform < bottom < all-k4 ordering holds).

7. **235B shows bimodal importance**: both early (0, 7, 21-27) and deep (85-92) layers
   are critical — middle layers (60-75) can be aggressively pruned.

8. **ffn_moe_out is the correct probe point.** l_out rankings are trivially depth-ordered
   (M1 vs M3 ρ > 0.98) due to residual stream accumulation.

---

## Files

| File | Description |
|------|-------------|
| `tools/imatrix-hsic/imatrix-hsic.cpp` | CKA probe: `--streaming-cka`, `--probe-l-out` flags |
| `tools/imatrix-hsic/CMakeLists.txt` | OpenMP linkage |
| `iccad/scripts/compute_cka.py` | Bootstrap (`--bootstrap N`), kernel-space optimization |
| `qwen3/cka-bootstrap.txt` | Qwen3-30B bootstrap results (200 resamples) |
| `qwen35/cka-bootstrap.txt` | Qwen3.5-35B bootstrap results (200 resamples) |

CKA probe data (chunk-averaged, ffn_moe_out):

| Directory | Model | Chunks |
|-----------|-------|--------|
| `data/qwen3-235b-hsic/` | Qwen3-235B | 840 |
| `data/qwen35-122b-hsic/` | Qwen3.5-122B | 829 |

Probe point comparison data (chunk-averaged, l_out/post_moe):

| Directory | Model | Chunks |
|-----------|-------|--------|
| `data/qwen3-30b-hsic-lout/` | Qwen3-30B | 840 |
| `data/qwen35-35b-hsic-lout/` | Qwen3.5-35B | 829 |
