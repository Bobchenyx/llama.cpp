# Router Analysis — Per-Layer Routing Metrics

## Method

During imatrix collection (E8 inference), the callback reads MoE router output logits
(`ffn_gate_inp`) and computes per-token routing statistics at two boundaries (k=4 and k=6,
both compared against k=8). Implemented in `tools/imatrix-iccad/imatrix-iccad.cpp`.

| Metric | k=4 definition | k=6 definition | Sensitivity signal |
|--------|---------------|---------------|-------------------|
| Score margin (SM) | prob[3] - prob[4] | prob[5] - prob[6] | Larger → safer to cut |
| Weight concentration (WC) | top4/top8 sum | top6/top8 sum | Higher → safer to cut |
| Max probability | max expert prob | (shared) | Higher → fewer experts needed |
| Routing entropy (RE) | -Σ p·log₂(p) | (shared) | Lower → safer to cut |

Calibration: ubergarm-imatrix-calibration-corpus-v02.txt (~430K tokens).

---

## Qwen3-30B (48 layers, 128 experts, no shared experts)

### SM vs ΔEntropy: uncorrelated (ρ = -0.07)

Score Margin (micro: per-token routing confidence under E8) and ΔEntropy (macro: expert
count distribution shift between E8/E4 runs) capture fundamentally different signals.
Their top-12 overlap is only 2/12 layers (12, 24).

### Weight distribution across top-8

| Segment | Share | Implication |
|---------|-------|-------------|
| Top 4 | 66.8% | Core — cutting k=8→4 loses 1/3 of weight |
| Experts 5-6 | 18.7% | Moderate contribution |
| Experts 7-8 | 14.5% | Marginal — k=6 is a natural intermediate |

### Layer 47 anomaly

Step-function routing: SM_k6 (0.0107) > SM_k4 (0.0101), the only layer where this occurs.
MaxProb = 0.127 (highest), RoutEntropy = 5.10 (lowest). Safest layer by every metric.

### Combined top-12 most sensitive (avg SM_k4+ΔEntropy+WC_k4+RE ranks)

Layers: **12, 24, 22, 23, 6, 10, 17, 11, 15, 21, 13, 25**

---

## Qwen3.5-35B (40 layers, 256 experts, with shared experts)

### Architecture difference

256 routed experts (vs 128) + shared experts with sigmoid gating:
`output = Σ(weight_i × expert_i(x)) + sigmoid(shared_gate(x)) × shared_expert(x)`

Absolute SM values are ~3x smaller than Qwen3 (more experts → thinner probability mass).
**Cross-model SM comparison requires normalization.**

### Monotonic depth gradient (all metrics agree)

| Layer group | SM_k4 avg | WC_k4 avg | Interpretation |
|-------------|-----------|-----------|----------------|
| 0-9 (early) | 0.00256 | 0.664 | More peaked — safest |
| 10-19 | 0.00249 | 0.653 | Moderate |
| 20-29 | 0.00223 | 0.650 | More uniform |
| 30-39 (deep) | 0.00196 | 0.639 | Most uniform — most sensitive |

Unlike Qwen3 (SM and ΔEntropy uncorrelated), all Qwen3.5 metrics correlate strongly
(SM↔WC ρ=+0.76, SM↔RE ρ=-0.85).

### Combined top-10 most sensitive

Layers: **38, 37, 35, 36, 31, 39, 27, 30, 26, 28**

### Shared expert compensation

Shared expert activation (`ffn_down_shexp` Σ(Act²)) varies dramatically: layer 38 = 2025
(critical) vs layer 23 = 9.67 (negligible). Layers where the shared expert is strong may
tolerate more aggressive routing reduction. This is confirmed by the PPL results: Qwen3.5's
all-k4 PPL gap (1.04) is smaller than Qwen3's (1.26) despite lower score margins.

---

## Qwen3.5-122B (48 layers, 256 experts, with shared experts)

### U-shaped sensitivity (not monotonic)

Unlike the 35B's monotonic gradient, the 122B has a U-shaped profile:

| Segment | SM_k4 avg | WC_k4 avg | Interpretation |
|---------|-----------|-----------|----------------|
| 0-11 (early) | 0.00180 | 0.655 | Low SM — routing uncertain |
| 12-23 (mid-early) | 0.00267 | 0.669 | Moderate — safest |
| 24-35 (mid-late) | 0.00345 | 0.668 | Highest SM — safest for pruning |
| 36-47 (deep) | 0.00296 | 0.643 | SM moderate, lowest WC |

SM↔WC correlation drops to ρ=+0.37 (vs 0.76 in 35B): routing confidence and weight
peakedness capture different sensitivity dimensions at this scale.

### Combined top-12 most sensitive

Layers: **3, 4, 0, 5, 46, 6, 2, 8, 47, 44, 7, 43**

Both shallow (0-8) and deep (43-47) are selected — qualitatively different from 35B.

---

## Cross-Model Comparison

| Aspect | Qwen3-30B | Qwen3.5-35B | Qwen3.5-122B |
|--------|-----------|-------------|--------------|
| Most sensitive | Early/mid (6,10,12) | Deep (35-39) | Early+deep (U-shape) |
| Safest | Deep (45-47) | Early (2,10,11) | Mid (24-35) |
| Depth trend | No pattern | Monotonic ↓ | U-shaped |
| SM↔WC corr | +0.57 | +0.76 | +0.37 |
| SM↔ΔEntropy | -0.07 | N/A | N/A |

**Key takeaway**: Sensitivity patterns are model-specific, not architecture-specific. Even
within the same family (Qwen3.5), scaling changes which layers are most sensitive. Per-model
analysis is required.

---

## Recommended Schedules (two-tier, k=8 vs k=4)

| Model | Keep k=8 | Count | Reduce to k=4 | FLOP saving |
|-------|----------|-------|----------------|-------------|
| Qwen3-30B | 6,10,11,12,13,15,17,21,22,23,24,25 | 12 | 36 layers | 37.5% |
| Qwen3.5-35B | 26,27,28,30,31,35,36,37,38,39 | 10 | 30 layers | 37.5% |
| Qwen3.5-122B | 0,2,3,4,5,6,7,8,43,44,46,47 | 12 | 36 layers | 37.5% |

Note: These are router-metric-based schedules. CKA M1-based schedules (validated by PPL)
are documented in [ppl-perlayer-topk.md](ppl-perlayer-topk.md).

---

## Data Sources

- Raw per-layer metrics: `qwen3-imatrix-q8_0-callback-864.txt`, `qwen35-imatrix-q8_0-callback-864.txt`, `qwen35-122B-imatrix-q8_0-callback-864.txt`
- Analysis script: `scripts/analyze_router_vs_entropy.py`
- Tool: `tools/imatrix-iccad/imatrix-iccad.cpp` (router analysis in callback)
