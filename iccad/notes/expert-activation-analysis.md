# Expert Activation Distribution Analysis

**Date**: 2026-03-17
**Model**: Qwen3 MoE (q8_0 imatrix)
**Data sources**:
- `qwen3-imatrix-q8_0-statistics-iccad.txt` — imatrix collected with `top_k=8` (E8)
- `qwen3-imatrix-q8_0-E4-statistics-iccad.txt` — imatrix collected with `top_k=4` (E4)

**Analysis script**: `analyze_expert_counts.py` (in workspace root)
**Plots**: `expert_activation_counts.png` (E8), `expert_activation_counts_E4.png` (E4)

---

## Setup

The model has **48 MoE layers**, each with **128 experts**. Under E8, each token activates 8 experts per layer:

```
total activations per layer (E8) = 3,440,640  →  ~430,080 tokens × 8
total activations per layer (E4) = 1,720,320  →  ~430,080 tokens × 4
```

All statistics in the `--show-statistics` output (Σ(Act²), entropy, ZD, etc.) are computed over the **normalized** values `e.values[ex][j] / e.counts[ex]`, i.e., average squared activation per expert invocation. The per-expert counts displayed in the new table are the raw unnormalized counts, which also serve as the normalization denominators.

---

## Key Findings

### 1. Expert load is highly non-uniform within each layer

Across both E8 and E4 runs, activation counts are far from uniform across the 128 experts per layer. Example extreme cases (E8):

| Layer | Hottest expert | Count | Coldest expert | Count | Hot/Cold ratio |
|-------|---------------|-------|---------------|-------|---------------|
| 14    | Exp55         | 181,665 | Exp71       | 1,342 | ~135×         |
| 38    | Exp55         | 217,394 | Exp117      | 3,532 | ~62×          |
| 43    | Exp1          | 1       | —            | —     | extreme       |
| 6     | Exp86         | 112,840 | Exp12       | 3,098 | ~36×          |

### 2. Σ(Act²) always increases under E4 — but differently by tensor type

Reducing top_k from 8→4 causes activation norms to **increase** (not decrease). This is expected: fewer experts → each selected expert compensates with stronger activations.

| Tensor | E4/E8 ratio range | Pattern |
|--------|--------------------|---------|
| `ffn_gate_exps` | 1.07–1.16 | Roughly uniform across all layers |
| `ffn_up_exps`   | 1.07–1.16 | Same as gate (tied routing) |
| `ffn_down_exps` | **1.10–1.60** | **Monotonically increasing with depth** |

The `ffn_down_exps` (output projection) ratio grows from ~1.1 at shallow layers to ~1.6 at layers 45–46. This suggests **later layers are most stressed by top_k reduction** and are strong candidates for retaining top_k=8.

### 3. Expert distribution concentration varies systematically across layers

Using the **Gini coefficient** and **Top-4 coverage %** (what fraction of total E8 activations come from the globally 4 most-used experts):

| Metric | Range | Most uniform | Most concentrated |
|--------|-------|-------------|------------------|
| Gini (E8) | 0.24–0.40 | Layers 0–1 | Layers 21, 25, 14 |
| Top-4% (E8) | 7–13% | Layers 0, 5 | Layers 14, 38, 17 |

Early layers (0–1) have the most uniform expert distributions — reducing top_k there loses the most diversity. Mid layers (14, 17, 24, 25, 38) are most concentrated — top_k=4 already captures a larger share of the mass there.

### 4. Entropy drops more in some layers than others under E4

ΔEntropy = Entropy(E4) − Entropy(E8), always negative. Largest drops (most routing collapse):

| Layer | ΔEntropy |
|-------|---------|
| 24    | −0.168  |
| 29    | −0.166  |
| 17    | −0.138  |
| 36    | −0.147  |
| 47    | −0.136  |

These layers see the most "forced concentration" under E4 — their routing diversity collapses the most when top_k is halved.

---

## Candidate Metrics for top_k Selection

The following per-layer metrics derived from imatrix data can guide which layers should keep top_k=8 vs. be reduced to top_k=4:

| Metric | Signal direction | How to use |
|--------|-----------------|-----------|
| **Gini(E8 counts)** | Higher → safer to prune | Layers with high Gini already route to a few experts; cutting top_k loses less |
| **Top-4 coverage % (E8)** | Higher → safer to prune | More of the mass is already captured by the top-4 experts |
| **ΔEntropy = Entr(E4)−Entr(E8)** | More negative → more sensitive | Layers where routing diversity collapses most under E4 |
| **ffn_down Σ(Act²) ratio (E4/E8)** | Higher → more sensitive | Late-layer output projections are most stressed; prefer keeping top_k=8 |
| **CosSim to adjacent layer (E8)** | Higher → safer to prune | Redundant layers can absorb more pruning |
| **ZD Score (E8)** | Higher → more important overall | Indicates presence of high-salience features; avoid pruning |

---

## Preliminary Layer Recommendations (Heuristic)

Based on the combined score `Top4%_E8 × ffn_down_ratio`, rough groupings:

**Candidates for top_k=4 (concentrated routing, robust gate/up activations)**:
- Layers 14, 17, 24, 25, 38 (high Top-4%, moderate ffn_down ratio)
- Layers 6, 7, 12, 13, 22, 23 (moderately concentrated)

**Candidates for top_k=8 (late layers, large ffn_down sensitivity)**:
- Layers 43–47 (ffn_down ratio ≥ 1.4; highest absolute Σ(Act²))
- Layers 40–42 (ffn_down ratio ~1.35–1.38)

**Ambiguous / needs perplexity validation**:
- Layers 0–5 (low absolute importance but very uniform routing)
- Layers 28–35 (mid-range on all metrics)

---

## Next Steps

1. Design a mixed top_k schedule (e.g., assign top_k=4 to 20–30 layers, top_k=8 to the rest) using the metrics above.
2. Evaluate perplexity of the two endpoints (all-E8, all-E4) as upper/lower bounds.
3. Run perplexity sweeps over candidate mixed schedules to validate metric guidance.
4. Cross-validate with expert quantization targets — layers important for pruning may need higher-precision quantization.
