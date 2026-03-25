# Qwen3.5-35B-A3B Router Analysis — Architecture Comparison with Qwen3

**Date**: 2026-03-24
**Model**: Qwen3.5-35B-A3B (Q8_0)
**Calibration**: ubergarm-imatrix-calibration-corpus-v02.txt (~424,448 tokens)
**Comparison**: Qwen3-30B-A3B-Instruct-2507 (Q8_0)

---

## Architecture Comparison

| Parameter | Qwen3 MoE | Qwen3.5 MoE |
|-----------|-----------|-------------|
| MoE layers | 48 | 40 |
| n_expert (routed) | 128 | **256** |
| n_expert_used | 8 | 8 |
| n_embd | 2048 | 2048 |
| n_ff (per routed expert) | 768 | **512** |
| **Shared experts** | **No** | **Yes** (n_ff_shexp = 512) |
| Graph model file | `qwen3moe.cpp` | `qwen35moe.cpp` |
| Total params | ~30B | ~35B |

**Key architectural difference**: Qwen3.5 has **shared experts** (`ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp`) plus a **shared expert gate** (`ffn_gate_inp_shexp`), in addition to 8 routed experts selected from 256. The MoE output is: `output = routed_moe_out + sigmoid(shared_gate) * shared_expert_out`.

`ffn_gate_inp` (shape `[256, 2048]`) produces 256 routing logits; `ffn_gate_inp_shexp` (shape `[1, 2048]`) produces a single sigmoid gate value controlling how much the shared expert contributes. They are **different weights** but receive the same input hidden state, which is why their Σ(Act²) values are identical. Our router analysis correctly captures the routing decisions over the 256 routed experts via `ffn_gate_inp`.

---

## Router Metric Comparison: Qwen3 vs Qwen3.5

| Metric | Qwen3 (128 experts) | Qwen3.5 (256 experts) | Ratio |
|--------|-------|---------|-------|
| SM_k4 range | 0.0054 – 0.0101 | **0.0014 – 0.0033** | ~0.33× |
| SM_k6 range | 0.0029 – 0.0107 | **0.0008 – 0.0018** | ~0.30× |
| WC_k4 range | 0.645 – 0.696 | **0.619 – 0.687** | slightly lower |
| WC_k6 range | 0.841 – 0.874 | **0.823 – 0.858** | slightly lower |
| MaxProb range | 0.081 – 0.127 | **0.023 – 0.063** | ~0.35× |
| RoutEntropy range | 5.10 – 6.28 | **7.28 – 7.73** | ~1.3× higher |

### Why the metrics differ

With 256 experts instead of 128, the probability mass is distributed over twice as many experts. Under a uniform distribution, each expert would get 1/256 ≈ 0.0039 instead of 1/128 ≈ 0.0078. This directly causes:

1. **Smaller score margins** — differences between adjacent probabilities are smaller (~3× smaller in practice)
2. **Lower max probability** — the top expert captures a smaller share (~3× smaller)
3. **Higher routing entropy** — more experts = higher baseline entropy (+1 bit for doubling experts in the uniform case)
4. **Slightly lower weight concentration** — top-k experts cover a smaller fraction of total probability

**This means absolute SM values between models are NOT directly comparable.** Cross-model comparison requires normalization (e.g., by the uniform-case expected gap, or by comparing relative rankings within each model).

---

## Qwen3.5 Per-Layer Router Metrics

| Layer | SM_k4 | SM_k6 | WC_k4 | WC_k6 | MaxProb | RoutEnt |
|-------|-------|-------|-------|-------|---------|---------|
| 0 | 0.002731 | 0.001474 | 0.651 | 0.841 | 0.041 | 7.393 |
| 1 | 0.002532 | 0.001274 | 0.657 | 0.843 | 0.043 | 7.486 |
| 2 | 0.002809 | 0.001337 | 0.687 | 0.858 | 0.063 | 7.374 |
| 3 | 0.001836 | 0.000878 | 0.655 | 0.840 | 0.041 | 7.586 |
| 4 | 0.002470 | 0.001193 | 0.661 | 0.845 | 0.042 | 7.512 |
| 5 | 0.002662 | 0.001285 | 0.666 | 0.848 | 0.045 | 7.489 |
| 6 | 0.002662 | 0.001344 | 0.660 | 0.845 | 0.045 | 7.472 |
| 7 | 0.002322 | 0.001136 | 0.662 | 0.845 | 0.044 | 7.519 |
| 8 | 0.002089 | 0.001015 | 0.660 | 0.843 | 0.041 | 7.580 |
| 9 | 0.002401 | 0.001176 | 0.666 | 0.847 | 0.043 | 7.516 |
| 10 | 0.003072 | 0.001570 | 0.660 | 0.845 | 0.049 | 7.346 |
| 11 | 0.003316 | 0.001770 | 0.661 | 0.847 | 0.053 | 7.277 |
| 12 | 0.002529 | 0.001325 | 0.657 | 0.843 | 0.048 | 7.420 |
| 13 | 0.002409 | 0.001267 | 0.652 | 0.841 | 0.044 | 7.459 |
| 14 | 0.002220 | 0.001173 | 0.648 | 0.838 | 0.043 | 7.492 |
| 15 | 0.002228 | 0.001174 | 0.647 | 0.838 | 0.042 | 7.486 |
| 16 | 0.002139 | 0.001138 | 0.644 | 0.836 | 0.041 | 7.492 |
| 17 | 0.002131 | 0.001110 | 0.646 | 0.837 | 0.040 | 7.516 |
| 18 | 0.002199 | 0.001163 | 0.649 | 0.839 | 0.041 | 7.508 |
| 19 | 0.002091 | 0.001066 | 0.649 | 0.838 | 0.040 | 7.534 |
| 20 | 0.002149 | 0.001018 | 0.662 | 0.844 | 0.042 | 7.572 |
| 21 | 0.002299 | 0.001141 | 0.662 | 0.845 | 0.043 | 7.528 |
| 22 | 0.002654 | 0.001347 | 0.652 | 0.841 | 0.045 | 7.402 |
| 23 | 0.002592 | 0.001360 | 0.653 | 0.841 | 0.046 | 7.370 |
| 24 | 0.002209 | 0.001167 | 0.649 | 0.838 | 0.043 | 7.462 |
| 25 | 0.002088 | 0.001081 | 0.643 | 0.835 | 0.039 | 7.507 |
| 26 | 0.002022 | 0.001053 | 0.643 | 0.835 | 0.040 | 7.521 |
| 27 | 0.002095 | 0.001058 | 0.641 | 0.834 | 0.037 | 7.522 |
| 28 | 0.002025 | 0.001041 | 0.642 | 0.835 | 0.039 | 7.507 |
| 29 | 0.002153 | 0.001091 | 0.647 | 0.838 | 0.037 | 7.548 |
| 30 | 0.002077 | 0.001048 | 0.646 | 0.837 | 0.036 | 7.567 |
| 31 | 0.001831 | 0.000939 | 0.641 | 0.834 | 0.033 | 7.614 |
| 32 | 0.002339 | 0.001201 | 0.649 | 0.840 | 0.037 | 7.498 |
| 33 | 0.002643 | 0.001423 | 0.655 | 0.845 | 0.036 | 7.501 |
| 34 | 0.002455 | 0.001321 | 0.648 | 0.841 | 0.034 | 7.522 |
| 35 | 0.001737 | 0.000831 | 0.639 | 0.833 | 0.028 | 7.706 |
| 36 | 0.001775 | 0.000912 | 0.639 | 0.834 | 0.028 | 7.681 |
| 37 | 0.001667 | 0.000900 | 0.630 | 0.830 | 0.025 | 7.703 |
| 38 | 0.001414 | 0.000774 | 0.619 | 0.823 | 0.023 | 7.733 |
| 39 | 0.001653 | 0.001448 | 0.626 | 0.832 | 0.027 | 7.648 |

---

## Qwen3.5 Routing Patterns

### Overall weight distribution

| Segment | Qwen3.5 | Qwen3 |
|---------|---------|-------|
| Top 4 / Top 8 (WC_k4) | **64.9%** | 66.8% |
| Top 6 / Top 8 (WC_k6) | **83.9%** | 85.5% |
| Experts 5-6 share | **19.0%** | 18.7% |
| Experts 7-8 share | **16.1%** | 14.5% |

Qwen3.5 distributes weight slightly more evenly across the top-8 experts than Qwen3. The marginal experts (7-8) contribute 16.1% vs 14.5% — they are slightly more important in Qwen3.5.

### Layer-depth pattern: Qwen3.5 has a monotonic gradient

Qwen3.5 shows a clear **monotonic trend**: deeper layers have smaller score margins, lower weight concentration, lower MaxProb, and higher routing entropy. This means **deeper layers route more uniformly** and are MORE sensitive to top_k reduction.

| Layer group | SM_k4 avg | WC_k4 avg | MaxProb avg | Interpretation |
|-------------|-----------|-----------|-------------|----------------|
| 0-9 (early) | 0.00256 | 0.664 | 0.044 | More peaked routing |
| 10-19 (mid-early) | 0.00249 | 0.653 | 0.044 | Moderate |
| 20-29 (mid-late) | 0.00223 | 0.650 | 0.040 | More uniform |
| 30-39 (deep) | 0.00196 | 0.639 | 0.030 | **Most uniform — most sensitive** |

This is the **opposite** of the ΔEntropy observation on Qwen3, where ΔEntropy flagged deep layers as sensitive but Score Margin said they were safe. In Qwen3.5, both perspectives agree: **deep layers are genuinely more sensitive**.

### Layer 39 anomaly

Layer 39 (last layer) has SM_k6 = 0.001448, which is very close to SM_k4 = 0.001653. The ratio SM_k6/SM_k4 = 0.876, the highest of all layers. Unlike Qwen3's layer 47 (where SM_k6 > SM_k4), Qwen3.5's last layer does not have a step-function profile, but it does have more uniform weight distribution at the k=6 boundary.

### No "Layer 47 anomaly" in Qwen3.5

Qwen3 had a dramatic outlier at layer 47 (SM_k4 = 0.0101, MaxProb = 0.127). Qwen3.5 has **no such outlier**. The most extreme layer (38) has SM_k4 = 0.001414, MaxProb = 0.023 — far more moderate. Qwen3.5's routing is uniformly flat across all layers.

---

## Per-Layer Sensitivity Ranking

Sensitivity is ranked by a **combined score** averaging 6 metric ranks: SM_k4, SM_k6, WC_k4, WC_k6, MaxProb, RoutEntropy. Rank 1 = most sensitive (risky to cut), rank 40 = safest. For SM/WC/MaxProb, smaller values = more sensitive; for RoutEntropy, larger values = more sensitive.

### Key finding: all metrics agree strongly

Unlike Qwen3 (where SM and ΔEntropy were uncorrelated at ρ = -0.07), Qwen3.5's router metrics are **highly correlated**:

| | SM_k4 | SM_k6 | WC_k4 | MaxProb | RoutEnt |
|---|---|---|---|---|---|
| SM_k4 | 1.00 | +0.85 | +0.76 | +0.78 | -0.85 |
| SM_k6 | +0.85 | 1.00 | +0.49 | +0.59 | -0.80 |
| WC_k4 | +0.76 | +0.49 | 1.00 | +0.80 | -0.50 |
| MaxProb | +0.78 | +0.59 | +0.80 | 1.00 | -0.79 |
| RoutEnt | -0.85 | -0.80 | -0.50 | -0.79 | 1.00 |

This is because Qwen3.5 has a **monotonic depth gradient** — all metrics move in the same direction with layer depth — so there is no SM-vs-DE style disagreement.

### Per-metric top 10 most sensitive layers

| Metric | Top 10 (most sensitive first) |
|--------|-------------------------------|
| SM_k4 | 38, 39, 37, 35, 36, 31, 3, 26, 28, 30 |
| SM_k6 | 38, 35, 3, 37, 36, 31, 8, 20, 28, 30 |
| WC_k4 | 38, 39, 37, 35, 36, 27, 31, 28, 25, 26 |
| MaxProb | 38, 37, 39, 35, 36, 31, 34, 30, 33, 32 |
| RoutEnt | 38, 35, 37, 36, 39, 31, 3, 8, 20, 30 |

**Deep layers (35-39) appear in every metric's top 10.** Layer 38 is rank 1 across all 5 metrics — the most sensitive layer by every measure.

### Combined ranking (full table, top 15)

| Rank | Layer | SM_k4 | SM_k6 | WC_k4 | MaxProb | RoutEnt | Combined |
|------|-------|-------|-------|-------|---------|---------|----------|
| 1 | 38 | 0.001414 | 0.000774 | 0.619 | 0.023 | 7.733 | 1.0 |
| 2 | 37 | 0.001667 | 0.000900 | 0.630 | 0.025 | 7.703 | 2.8 |
| 3 | 35 | 0.001737 | 0.000831 | 0.639 | 0.028 | 7.706 | 3.3 |
| 4 | 36 | 0.001775 | 0.000912 | 0.639 | 0.028 | 7.681 | 5.2 |
| 5 | 31 | 0.001831 | 0.000939 | 0.641 | 0.033 | 7.614 | 6.2 |
| 6 | 39 | 0.001653 | 0.001448 | 0.626 | 0.027 | 7.648 | 8.7 |
| 7 | 27 | 0.002095 | 0.001058 | 0.641 | 0.037 | 7.522 | 10.3 |
| 8 | 30 | 0.002077 | 0.001048 | 0.646 | 0.036 | 7.567 | 10.7 |
| 9 | 26 | 0.002022 | 0.001053 | 0.643 | 0.040 | 7.521 | 11.5 |
| 10 | 28 | 0.002025 | 0.001041 | 0.642 | 0.039 | 7.507 | 11.8 |
| 11 | 25 | 0.002088 | 0.001081 | 0.643 | 0.039 | 7.507 | 13.5 |
| 12 | 3 | 0.001836 | 0.000878 | 0.655 | 0.041 | 7.586 | 14.2 |
| 13 | 29 | 0.002153 | 0.001091 | 0.647 | 0.037 | 7.548 | 14.7 |
| 14 | 19 | 0.002091 | 0.001066 | 0.649 | 0.040 | 7.534 | 14.8 |
| 15 | 17 | 0.002131 | 0.001110 | 0.646 | 0.040 | 7.516 | 15.0 |

### Layer 39 anomaly

Layer 39 has a split personality: SM_k4 rank = 2 (very sensitive for k=4 cut) but SM_k6 rank = 37 (very safe for k=6 cut). Its SM_k6 = 0.001448 is close to early-layer values, while SM_k4 = 0.001653 is among the lowest. This suggests a steep drop between expert positions 4-5 but a **flat** profile around positions 6-7 — the opposite of a step function. For this layer, k=6 may be viable even though k=4 is risky.

---

## Recommended Schedules

### Three-tier schedule (k=8 / k=6 / k=4)

| Tier | top_k | Layers | Count |
|------|-------|--------|-------|
| 1 | 8 | 26, 27, 28, 30, 31, 35, 36, 37, 38, 39 | 10 |
| 2 | 6 | 3, 8, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 29, 32, 34 | 15 |
| 3 | 4 | 0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 22, 23, 33 | 15 |

**Computation saving**: 10 layers at full cost + 15 layers at 75% + 15 layers at 50% = **32.5% MoE expert FLOP reduction** vs all-E8.

### Two-tier schedule (k=8 vs k=4)

| Tier | top_k | Layers | Count |
|------|-------|--------|-------|
| 1 | 8 | 26, 27, 28, 30, 31, 35, 36, 37, 38, 39 | 10 |
| 2 | 4 | 0–25, 29, 32, 33, 34 | 30 |

**Computation saving**: 10 layers at full cost + 30 layers at 50% = **37.5% MoE expert FLOP reduction** vs all-E8.

### Contrast with Qwen3

| Aspect | Qwen3 | Qwen3.5 |
|--------|-------|---------|
| Most sensitive layers | Early/mid (6, 10, 11, 12) | **Deep** (35, 36, 37, 38, 39) |
| Safest layers | Deep (45, 46, 47) | **Early** (2, 10, 11) |
| Depth-sensitivity pattern | No clear trend | **Monotonic**: deeper = more sensitive |
| Metric agreement | SM and DE uncorrelated (ρ=-0.07) | All metrics agree (ρ=0.49-0.85) |

Qwen3.5's monotonic depth gradient makes schedule design simpler: the decision boundary is essentially "how deep is the layer?" In Qwen3, sensitivity was scattered across layers with no depth pattern.

---

## Implications for Top_k Reduction

### Qwen3.5 is harder to prune than Qwen3

| Factor | Qwen3 | Qwen3.5 | Impact |
|--------|-------|---------|--------|
| Score margins | 0.005-0.010 | 0.001-0.003 | Router is **less confident** in Qwen3.5 → more risky to cut |
| MaxProb | 0.08-0.13 | 0.02-0.06 | No dominant expert → every expert matters more |
| Weight concentration | 0.65-0.70 | 0.62-0.69 | Top-4 captures less weight → cutting loses more |
| Shared experts | None | Gated, always active | Shared expert provides a **safety net** that may mitigate top_k reduction |

The routing in Qwen3.5 is fundamentally more **spread out** (256 experts, lower concentration). However, the presence of **shared experts** — which always contribute regardless of routing — may provide a buffer that compensates for lost routed experts. This is a critical factor that our current per-token router metrics do **not** capture.

### Shared expert compensation hypothesis

In Qwen3.5, the MoE output is:
```
output = Σ(weight_i × expert_i(x)) + sigmoid(shared_gate(x)) × shared_expert(x)
```

The shared expert contribution is **gated per-token** by a sigmoid scalar. If the gate is close to 1 and the shared expert captures the "average" behavior, then the routed experts only need to capture the **residual**. In this case, reducing top_k would lose less information than the router metrics alone suggest, because the shared expert already provides baseline coverage. The per-layer sigmoid gate value distribution (how often near 0 vs near 1) would indicate how much the shared expert actually contributes.

To test this hypothesis: compare perplexity degradation under k=4 for Qwen3 vs Qwen3.5. If Qwen3.5 degrades less despite lower score margins, the shared expert is providing compensation.

---

## Tool Compatibility Assessment

| Feature | Status | Notes |
|---------|--------|-------|
| Router analysis (SM, WC, MaxProb, RE) | **Working** | Correctly reads `ffn_gate_inp` output with 256 experts |
| Per-expert activation counts | **Working** | Shows 256 experts per layer correctly |
| GGUF save/load | **Working** | Format is model-agnostic |
| Shared expert tracking | **Not implemented** | `ffn_gate_inp_shexp` (shape `[1, 2048]`) is a separate sigmoid gate, not analyzed; its Σ(Act²) matches `ffn_gate_inp` because they share the same input |
| Shared expert FFN importance | **Not tracked** | `ffn_down_shexp` Σ(Act²) values show shared expert contribution but no dedicated analysis |

### Potential enhancement

The shared expert's activation magnitude (`ffn_down_shexp` Σ(Act²)) varies dramatically across layers:
- Layer 38: Σ(Act²) = 2025 (very high — shared expert is critical)
- Layer 23: Σ(Act²) = 9.67 (very low — shared expert barely contributes)

This could serve as an additional metric: layers where the shared expert contributes strongly may tolerate more aggressive top_k reduction on the routed experts.

---

## Next Steps

1. **Perplexity comparison**: Run all-E4 perplexity for Qwen3.5 and compare degradation vs Qwen3 to test shared expert compensation hypothesis.
2. **Normalize metrics for cross-model comparison**: Define a metric normalization scheme (e.g., SM / expected_uniform_gap) to enable fair comparison between 128-expert and 256-expert models.
3. **Shared expert importance analysis**: Correlate `ffn_down_shexp` Σ(Act²) with per-layer routing metrics to determine whether shared expert strength predicts tolerance to top_k reduction.
4. **ΔEntropy for Qwen3.5**: Run E4 imatrix to compute ΔEntropy and test whether the SM-DE uncorrelation pattern holds for this architecture.
