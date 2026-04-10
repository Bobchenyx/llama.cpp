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

## Conclusions

1. **CKA implementation correct.** Python and C++ match the standard linear CKA formula.

2. **Bootstrap: top 5-7 layers robust**, positions 8-12 volatile but with similar M1 values.

3. **Chunk averaging changes rankings** (ρ ≈ 0.6, 58-67% top-12 overlap). Top-5 consistent.

4. **Layer 46 (Qwen3) is a chunk-averaging artifact** — rank 1 chunk, rank ~28 token-level.

5. **Token-level PPL is slightly better** (ΔPPL = -0.03 to -0.05). Both methods valid;
   token-level recommended when compute allows.

---

## Files

| File | Description |
|------|-------------|
| `tools/imatrix-hsic/imatrix-hsic.cpp` | StreamingCKA mode (`--streaming-cka`) |
| `tools/imatrix-hsic/CMakeLists.txt` | OpenMP linkage |
| `iccad/scripts/compute_cka.py` | Bootstrap (`--bootstrap N`), kernel-space optimization |
| `qwen3/cka-bootstrap.txt` | Qwen3 bootstrap results (200 resamples) |
| `qwen35/cka-bootstrap.txt` | Qwen3.5 bootstrap results (200 resamples) |
