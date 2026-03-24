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

Using the **Gini coefficient** of per-expert activation counts:

| Metric | Range | Most uniform | Most concentrated |
|--------|-------|-------------|------------------|
| Gini (E8) | 0.24–0.40 | Layers 0–1 | Layers 21, 25, 14 |

Early layers (0–1) have the most uniform expert distributions — reducing top_k there loses the most diversity. Mid layers (14, 17, 24, 25) are most concentrated.

> **Note**: "Top-K coverage %" (fraction of total activations from the globally K most-used experts) was evaluated and **discarded** as a metric — it measures global popularity, but routing is per-token and dynamic, so this statistic has no practical meaning for top_k selection.

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
| **ΔEntropy = Entr(E4)−Entr(E8)** | More negative → more sensitive | Layers where routing diversity collapses most under E4 |
| **ffn_down Σ(Act²) ratio (E4/E8)** | Reference only | Magnitude scaling, correctable by a scaling factor; not a standalone criterion |
| **CosSim to adjacent layer (E8)** | Higher → safer to prune | Redundant layers can absorb more pruning |
| **ZD Score (E8)** | Higher → more important overall | Indicates presence of high-salience features; avoid pruning |

> **Discarded**: Top-K coverage % — global expert popularity is meaningless because routing is per-token dynamic selection.

---

## Preliminary Layer Recommendations (Heuristic)

Based on **ΔEntropy** as the primary sensitivity metric (most routing diversity collapse = should keep top_k=8):

**Top 12 layers to keep at top_k=8** (by ΔEntropy, most sensitive first):
Layers 24, 29, 37, 36, 12, 17, 47, 35, 33, 19, 34, 32

**Remaining 36 layers use top_k=4**:
Layers 0–11, 13–16, 18, 20–23, 25–28, 30–31, 38–46

Note: these are distributed across the full depth, not concentrated in late layers — routing diversity loss is not monotonically correlated with depth.

---

## Next Steps

1. Design a mixed top_k schedule (e.g., assign top_k=4 to 20–30 layers, top_k=8 to the rest) using the metrics above.
2. Evaluate perplexity of the two endpoints (all-E8, all-E4) as upper/lower bounds.
3. Run perplexity sweeps over candidate mixed schedules to validate metric guidance.
4. Cross-validate with expert quantization targets — layers important for pruning may need higher-precision quantization.
