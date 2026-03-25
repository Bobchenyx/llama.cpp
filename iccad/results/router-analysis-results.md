# Router Analysis Results — Cross-Comparison with ΔEntropy

**Date**: 2026-03-23 (updated 2026-03-24 with k=6 data)
**Model**: Qwen3-30B-A3B-Instruct-2507 (Q8_0)
**Calibration**: ubergarm-imatrix-calibration-corpus-v02.txt (~430,080 tokens, ~840 chunks)
**Analysis script**: `scripts/analyze_router_vs_entropy.py`

---

## Data Sources

| Metric | Source | Method |
|--------|--------|--------|
| Score Margin (k=4) | Router analysis (Phase 1a) | Per-token softmax → sort → prob[3]-prob[4] |
| Score Margin (k=6) | Router analysis (Phase 1a) | Per-token softmax → sort → prob[5]-prob[6] |
| Weight Conc. (k=4) | Router analysis (Phase 1a) | top4_sum / top8_sum per token |
| Weight Conc. (k=6) | Router analysis (Phase 1a) | top6_sum / top8_sum per token |
| Routing Entropy | Router analysis (Phase 1a) | -Σ p·log₂(p) over 128 experts per token |
| ΔEntropy | E8/E4 expert count comparison | Shannon entropy of per-expert activation counts, E4 − E8 |

All router metrics are collected during a single E8 inference pass. ΔEntropy requires two separate runs (E8 and E4).

---

## Key Finding 1: Score Margin and ΔEntropy Are Uncorrelated

**Spearman rank correlation matrix (6 metrics):**

| | SM_k4 | SM_k6 | WC_k4 | WC_k6 | DE | RE |
|---|---|---|---|---|---|---|
| SM_k4 | 1.00 | +0.92 | +0.57 | +0.77 | **-0.07** | +0.79 |
| SM_k6 | +0.92 | 1.00 | +0.28 | +0.53 | **-0.12** | +0.90 |
| WC_k4 | +0.57 | +0.28 | 1.00 | +0.92 | +0.25 | +0.08 |
| WC_k6 | +0.77 | +0.53 | +0.92 | 1.00 | +0.15 | +0.32 |
| DE | -0.07 | -0.12 | +0.25 | +0.15 | 1.00 | -0.18 |
| RE | +0.79 | +0.90 | +0.08 | +0.32 | -0.18 | 1.00 |

**SM and DE are essentially uncorrelated (ρ = -0.07 at k=4, ρ = -0.12 at k=6).** The two metrics capture fundamentally different signals:

- **Score Margin** (micro-level): per-token routing confidence at the boundary, measured under the true E8 input distribution.
- **ΔEntropy** (macro-level): how much the overall expert usage distribution shifts when top_k is globally changed. Subject to inter-layer distribution shift.

---

## Key Finding 2: k=4 and k=6 Rankings Are Highly Consistent

SM_k4 and SM_k6 rankings correlate at **ρ = +0.92**. WC_k4 and WC_k6 correlate at **ρ = +0.92**.

| Metric | Top 12 most sensitive layers |
|--------|-------------------------------|
| SM_k4 | 6, 12, 22, 10, 14, 11, 23, 24, 9, 1, 21, 15 |
| SM_k6 | 1, 6, 3, 12, 10, 14, 7, 11, 9, 15, 22, 5 |

**Overlap: 9/12 layers** (1, 6, 9, 10, 11, 12, 14, 15, 22). The core set of sensitive layers is stable across boundaries, confirming that router metrics are reliable.

---

## Key Finding 3: Weight Distribution Across Top-8 Experts

| Segment | Share of top-8 weight | Implication |
|---------|----------------------|-------------|
| Top 4 experts | **66.8%** | Core routing — cutting loses 1/3 of weight |
| Experts 5-6 | **18.7%** | Significant but moderate contribution |
| Experts 7-8 | **14.5%** | Marginal — cutting loses the least |

This suggests **k=6 is a natural intermediate point**: reducing from k=8 to k=6 loses only ~14.5% of the top-8 weight while saving 25% of expert computation. Reducing further from k=6 to k=4 is a larger step, losing another ~18.7%.

---

## Key Finding 4: Layer 47 Anomaly

Layer 47 exhibits a unique **step-function routing profile**:

| Metric | Value | Rank (out of 48) |
|--------|-------|-----------------|
| SM_k4 | 0.0101 | 48 (safest) |
| SM_k6 | **0.0107** | 48 (safest) |
| MaxProb | 0.127 | 1 (highest) |
| RoutEntropy | 5.10 | 48 (lowest) |

**SM_k6 > SM_k4**: the gap between experts 6-7 (0.0107) is larger than the gap between experts 4-5 (0.0101). This is the only layer where this occurs, indicating a bimodal routing pattern — a few experts are strongly preferred, then there is a cliff around position 6-7. Layer 47 is consistently the safest to reduce across all router metrics.

---

## Top 12 Sensitive Layers by Each Metric

| Metric | Top 12 (most sensitive first) |
|--------|-------------------------------|
| SM_k4 | 6, 12, 22, 10, 14, 11, 23, 24, 9, 1, 21, 15 |
| SM_k6 | 1, 6, 3, 12, 10, 14, 7, 11, 9, 15, 22, 5 |
| ΔEntropy | 24, 29, 37, 36, 12, 17, 47, 35, 33, 19, 34, 32 |
| WC_k4 | 22, 25, 23, 24, 21, 26, 13, 27, 12, 10, 34, 11 |
| Routing Entropy | 1, 3, 0, 6, 2, 12, 4, 10, 11, 5, 7, 9 |
| **Combined (avg SM_k4+DE+WC_k4+RE)** | **12, 24, 22, 23, 6, 10, 17, 11, 15, 21, 13, 25** |

**Overlap between SM_k4 and DE top 12: only 2 layers (12, 24).**

---

## Disagreement Analysis

Largest disagreements between Score Margin rank and ΔEntropy rank:

| Layer | SM Rank | DE Rank | |Diff| | Interpretation |
|-------|---------|---------|--------|----------------|
| 37 | 45 | 3 | 42 | DE says very sensitive, SM says safe — deep layer with confident routing but large distributional shift |
| 47 | 48 | 7 | 41 | Same pattern: DE sensitive, SM safe — last layer, step-function routing |
| 14 | 5 | 45 | 40 | SM says very sensitive, DE says safe — uncertain routing but stable global distribution |
| 9 | 9 | 46 | 37 | Same pattern: SM sensitive, DE safe |
| 6 | 1 | 34 | 33 | SM: most sensitive layer; DE: rank 34 |

**Hypothesis**: ΔEntropy's sensitivity ranking for deep layers may be inflated by cascading distribution shift — when ALL layers use E4, the cumulative effect of upstream changes distorts the input distribution at deep layers. Score Margin, measured under the true E8 distribution, may be more reliable for isolating per-layer sensitivity.

---

## Recommended Schedules (Preliminary)

### Two-tier schedule (k=8 vs k=4)

Combined ranking (average of SM_k4, DE, WC_k4, RE ranks):

- Keep top_k=8 (12 layers): **6, 10, 11, 12, 13, 15, 17, 21, 22, 23, 24, 25**
- Reduce to top_k=4 (36 layers): all others

### Three-tier schedule (k=8 / k=6 / k=4)

Based on SM_k4 ranking (most sensitive → least sensitive):

| Tier | top_k | Layers | Count | Weight retained |
|------|-------|--------|-------|-----------------|
| 1 | 8 | 1, 6, 9, 10, 11, 12, 14, 15, 21, 22, 23, 24 | 12 | 100% |
| 2 | 6 | 0, 3, 4, 5, 7, 8, 13, 16, 17, 18, 19, 20, 25, 26, 27, 28, 29, 30, 33, 34 | 20 | ~85.4% |
| 3 | 4 | 2, 31, 32, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 | 16 | ~67.5% |

The three-tier schedule offers a more gradual degradation path than the binary two-tier approach. Tier 2 (k=6) preserves 85.4% of the top-8 expert weight while reducing computation by 25%.

---

## Next Steps

1. **Phase 2 validation**: Run per-layer perplexity experiments for the most disputed layers (37, 47, 6, 14, 9) to determine which metric better predicts perplexity impact.
2. **Three-tier perplexity evaluation**: Compare perplexity of the three-tier schedule vs. the two-tier schedule vs. uniform k=4.
3. **Qwen3.5 cross-validation**: Run the same router analysis on Qwen3.5-35B-A3B (which has shared experts) to verify tool applicability and compare routing patterns.
