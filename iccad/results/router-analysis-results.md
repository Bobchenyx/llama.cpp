# Router Analysis Results — Cross-Comparison with ΔEntropy

**Date**: 2026-03-23
**Model**: Qwen3-30B-A3B-Instruct-2507 (Q8_0)
**Calibration**: ubergarm-imatrix-calibration-corpus-v02.txt (~430,080 tokens, ~840 chunks)
**Analysis script**: `analyze_router_vs_entropy.py` (workspace root)

---

## Data Sources

| Metric | Source | Method |
|--------|--------|--------|
| Score Margin | Router analysis (Phase 1a) | Per-token softmax → sort → prob[3]-prob[4], collected during E8 inference |
| Weight Concentration | Router analysis (Phase 1a) | top4_sum / top8_sum per token |
| Routing Entropy | Router analysis (Phase 1a) | -Σ p·log₂(p) over 128 experts per token |
| ΔEntropy | E8/E4 expert count comparison | Shannon entropy of per-expert activation counts, E4 − E8 |

---

## Key Finding: Score Margin and ΔEntropy Are Uncorrelated

**Spearman rank correlation matrix:**

| | SM | DE | WC | RE |
|---|---|---|---|---|
| Score Margin (SM) | 1.00 | **-0.07** | +0.57 | +0.79 |
| ΔEntropy (DE) | -0.07 | 1.00 | +0.25 | -0.18 |
| Weight Conc. (WC) | +0.57 | +0.25 | 1.00 | +0.08 |
| Routing Entropy (RE) | +0.79 | -0.18 | +0.08 | 1.00 |

**SM and DE are essentially uncorrelated (ρ = -0.07).** This confirms the hypothesis from the design document: the two metrics capture fundamentally different signals.

- **Score Margin** (micro-level): per-token routing confidence at the k=4 boundary, measured under the true E8 input distribution. Tells us how "close" the 5th expert is to the 4th at each decision point.
- **ΔEntropy** (macro-level): how much the overall expert usage distribution shifts when top_k is globally changed from 8 to 4. Subject to inter-layer distribution shift because all layers are changed simultaneously.

The router metrics (SM, WC, RE) correlate with each other (all measure per-token routing behavior), but do not correlate with ΔEntropy (which measures a system-level distributional shift).

---

## Top 12 Sensitive Layers by Each Metric

| Metric | Top 12 (most sensitive first) |
|--------|-------------------------------|
| Score Margin | 6, 12, 22, 10, 14, 11, 23, 24, 9, 1, 21, 15 |
| ΔEntropy | 24, 29, 37, 36, 12, 17, 47, 35, 33, 19, 34, 32 |
| Weight Conc. | 22, 25, 23, 24, 21, 26, 13, 27, 12, 10, 34, 11 |
| Routing Entropy | 1, 3, 0, 6, 2, 12, 4, 10, 11, 5, 7, 9 |
| **Combined (avg rank)** | **12, 24, 22, 23, 6, 10, 17, 11, 15, 21, 13, 25** |

**Overlap between SM and DE top 12: only 2 layers (12, 24).** Out of 12 candidates from each metric, only layers 12 and 24 are identified as sensitive by both. This is far below what would be expected from correlated metrics.

---

## Disagreement Analysis

Largest disagreements between Score Margin rank and ΔEntropy rank:

| Layer | SM Rank | DE Rank | |Diff| | Interpretation |
|-------|---------|---------|--------|----------------|
| 37 | 45 | 3 | 42 | DE says very sensitive, SM says safe — deep layer with confident routing but large distributional shift |
| 47 | 48 | 7 | 41 | Same pattern: DE sensitive, SM safe — last layer, highest max_prob (0.127) |
| 14 | 5 | 45 | 40 | SM says very sensitive, DE says safe — uncertain routing but stable global distribution |
| 9 | 9 | 46 | 37 | Same pattern: SM sensitive, DE safe |
| 6 | 1 | 34 | 33 | SM: most sensitive layer; DE: rank 34 |

**Pattern**: Deep layers (37, 47, 35, 36) have **large score margins** (confident routing) but **large ΔEntropy** (distribution shift under E4). Early/mid layers (6, 9, 10, 14) have **small score margins** (uncertain routing) but **small ΔEntropy** (stable distribution).

**Hypothesis**: ΔEntropy's sensitivity ranking for deep layers may be inflated by cascading distribution shift — when ALL layers use E4, the cumulative effect of upstream changes distorts the input distribution at deep layers, causing exaggerated routing changes. Score Margin, measured under the true E8 distribution, may be more reliable for these layers.

---

## Router Metric Ranges

| Layer | ScoreMargin | WeightConc | MaxProb | RoutEntropy |
|-------|-------------|------------|---------|-------------|
| Min | 0.0054 (L6) | 0.645 (L22) | 0.081 (L10) | 5.10 (L47) |
| Max | 0.0101 (L47) | 0.696 (L0) | 0.127 (L47) | 6.28 (L1) |
| Mean | 0.0069 | 0.667 | 0.094 | 5.90 |

Overall range is narrow (SM spans 0.0054–0.0101, a ~1.87× ratio), suggesting that most layers have similar routing confidence. Layer 47 is a clear outlier — it has the largest margin, highest max probability, and lowest entropy, consistent with a highly peaked "cleanup/finalization" layer.

---

## Recommended Schedule (Preliminary)

**Combined ranking (average of SM, DE, WC, RE ranks):**

Keep top_k=8 (12 layers): **6, 10, 11, 12, 13, 15, 17, 21, 22, 23, 24, 25**

Reduce to top_k=4 (36 layers): 0–5, 7–9, 14, 16, 18–20, 26–47

**Note**: This is a preliminary recommendation based on combined ranking. The combined ranking favors mid-layers (10–25) which appear in multiple metrics' sensitive lists. Deep layers (37, 47) that ΔEntropy flags are excluded because the router metrics (SM, WC, RE) all consider them safe.

**Validation required**: Per-layer perplexity experiments (Phase 2, Approach B) are needed to determine which metric better predicts actual quality impact. If perplexity correlates more with SM than DE, the combined schedule above is appropriate. If perplexity correlates with DE, the deep layers should be kept at top_k=8.

---

## Next Steps

1. **Phase 2 validation**: Run per-layer perplexity experiments for the most disputed layers (37, 47, 6, 14, 9) to determine which metric better predicts perplexity impact.
2. **Budget sweep**: Test schedules with different numbers of E8 layers (8, 12, 16, 24) to find the Pareto frontier of quality vs. efficiency.
3. **Consider multiple combined rankings**: weighted average (e.g., heavier weight on SM if validated), Borda count, or other rank aggregation methods.
