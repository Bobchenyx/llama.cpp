# Per-Layer top_k Sensitivity Analysis — Design Document

**Date**: 2026-03-23
**Status**: Phase 1a implemented (router-level analysis in callback)

---

## Motivation

Our current analysis compares two "pure" configurations (all-E8 vs all-E4) and uses ΔEntropy to rank layer sensitivity. However, the actual deployment target is a **mixed top_k schedule** — some layers at top_k=8, others at top_k=4. This introduces a **distribution shift problem**: changing top_k at one layer alters the input distribution for all downstream layers.

Metrics derived from comparing pure E8 vs pure E4 may not accurately predict per-layer sensitivity in a mixed configuration. We need a method that isolates each layer's sensitivity independently.

---

## Goal

For each of the 48 MoE layers, measure **how much the layer's output changes** when its top_k is reduced (e.g., 8→4), while keeping all other layers at their baseline configuration (top_k=8). This gives a per-layer sensitivity score free from inter-layer distribution shift.

---

## Methodology Overview

### What the baseline imatrix tool does

The standard imatrix tool collects **squared input activation norms** (`Σ(x²)`) for every weight tensor during calibration inference. For MoE layers, each expert is counted separately. The resulting importance matrix guides quantization — dimensions with larger activations receive higher precision.

### What we add: router-level analysis

During the same inference pass, when the callback detects a MoE router weight tensor (`ffn_gate_inp`), we additionally read the **output tensor** (router logits `[n_expert, n_tokens]`) and compute per-token routing statistics. This is zero-cost relative to the existing imatrix collection — it only requires one additional GPU→CPU copy of a small tensor (128 × n_tokens floats) per MoE layer.

For each token, we compute softmax over all 128 expert logits, sort descending, then extract metrics at **two boundaries** (k=4 and k=6, both compared against k=8):

| Metric | k=4 definition | k=6 definition | What it tells us |
|--------|---------------|---------------|-----------------|
| Score margin | `prob[3] - prob[4]` | `prob[5] - prob[6]` | How confident the router is at the boundary. Large gap → safe to cut. |
| Weight concentration | `top4_sum / top8_sum` | `top6_sum / top8_sum` | How much of the top-8 weight is carried by the top-k. High → safe to cut. |
| Max probability | max expert probability | (shared) | Whether routing is dominated by a single expert. High → fewer experts needed. |
| Routing entropy | `-Σ p·log₂(p)` over all experts | (shared) | Overall spread of the routing distribution. Low → peaked → fewer experts needed. |

### How this relates to the earlier ΔEntropy analysis

We previously compared two full-model runs (all-E8 vs all-E4) and used **ΔEntropy of expert activation counts** to rank layer sensitivity. That approach measures a **macro-level** effect: how much the overall expert usage distribution changes when top_k is halved.

Router analysis measures a **micro-level** signal: at each individual token, how much does the router "want" to select more than 4 experts? The two perspectives are complementary:

- **ΔEntropy** captures aggregate distribution changes across the full calibration set, but suffers from distribution shift (all layers are changed simultaneously).
- **Score margin** captures per-token routing confidence at the k=4 boundary under the true E8 input distribution, free from distribution shift.

If both metrics agree (a layer has large ΔEntropy AND small score margin), the layer is clearly sensitive. If they disagree, further validation is needed (Phase 1b offline replay or Phase 2 perplexity experiments).

### Decision procedure (planned)

Once router analysis data is collected:

1. Rank all 48 layers by mean score margin (ascending = most sensitive first).
2. Cross-reference with ΔEntropy ranking from the earlier E8/E4 analysis.
3. Select a budget (e.g., 12 layers keep top_k=8, 36 layers reduce to top_k=4).
4. Assign top_k=8 to the most sensitive layers by combined ranking.
5. Validate the mixed schedule via perplexity evaluation.

---

## Proposed Approaches

### Approach A: Save MLP inputs, offline replay (Recommended)

**Idea**: Run one full E8 forward pass, save the hidden state entering each MoE block. Then for each layer independently, replay the saved input through that layer's MoE computation with different top_k settings and compare outputs.

**Steps**:

1. **Collect phase** (one forward pass through the model):
   - Modify imatrix-iccad callback to also capture the hidden state tensor **before** each MoE block (i.e., the input to the router + experts).
   - Save per-layer inputs to disk (one file per layer, or a single HDF5/numpy archive).
   - Data size estimate: 48 layers × ~430K tokens × hidden_dim × float16 — may need chunking or sampling.

2. **Replay phase** (offline, per-layer, no full model inference):
   - Load the saved hidden states for layer L.
   - Load layer L's weights from GGUF: router (gate) weights, expert FFN weights (gate_exps, up_exps, down_exps).
   - Compute MoE forward pass with top_k=8 → output_E8.
   - Compute MoE forward pass with top_k=4 → output_E4.
   - Measure difference: cosine similarity, L2 distance, relative norm change, etc.

3. **Result**: per-layer sensitivity score based on actual output difference, not routing statistics.

**Pros**:
- Only one full model inference required; all per-layer experiments are independent matrix ops
- No distribution shift — each layer sees its real E8 input
- Fast iteration on metrics (can try different top_k values, different distance measures)
- Can be implemented in Python (numpy/torch) for rapid prototyping

**Cons**:
- Requires saving large intermediate activation tensors (memory/disk pressure)
- Need to implement MoE forward pass outside of llama.cpp (router softmax → top_k → expert matmul → weighted sum)
- Doesn't capture cascading effects (how one layer's change affects downstream layers)

**Implementation considerations**:
- Hidden state capture: the imatrix callback already fires on `GGML_OP_MUL_MAT_ID` and has access to `src1` (input activations). We need to also capture the **pre-routing** hidden state — this is the input to the gate/router linear layer, which is a `GGML_OP_MUL_MAT` node with the gate weight tensor.
- Memory management: for Qwen3 MoE with ~430K tokens and hidden_dim=2048 (or larger), saving all tokens for all layers may require significant disk space. Consider sampling a subset of tokens or processing in chunks.
- Router weight access: need to extract the gate weight matrix from the GGUF file to compute router logits independently.

---

### Approach B: Per-layer top_k in llama.cpp inference

**Idea**: Modify llama.cpp to accept a per-layer `num_experts_per_tok` schedule, then run N perplexity experiments — each with only one layer changed to E4, all others at E8.

**Steps**:

1. Modify `src/llama-graph.cpp` (or model config) to support a per-layer top_k array instead of a single global value.
2. Run 48 perplexity evaluations, each with configuration: "layer L = top_k=4, all others = top_k=8".
3. Compare each perplexity to baseline (all-E8) → per-layer perplexity delta.

**Pros**:
- Most accurate end-to-end measure (includes cascading effects through all downstream layers)
- Directly measures perplexity impact, which is the ultimate evaluation criterion

**Cons**:
- Requires modifying core llama.cpp inference code (llama-graph, model config parsing)
- 48 full perplexity runs — expensive (hours of GPU time per run)
- Harder to iterate on quickly

**Implementation considerations**:
- The per-layer top_k schedule could be passed via a JSON config file or command-line parameter.
- Key code location: `src/llama-graph.cpp` where `n_expert_used` is read from model hyperparameters to construct the MoE graph nodes.
- Could parallelize by running multiple perplexity evaluations simultaneously if GPU memory allows.

---

### Approach C: Capture router logits via callback

**Idea**: During E8 forward pass, hook the router's `GGML_OP_MUL_MAT` to capture the full 128-dim logit vector for each token at each layer. Then offline, simulate different top_k selections and estimate output difference from logit distributions.

**Steps**:

1. Identify the router gate tensor name pattern (e.g., `blk.{L}.ffn_gate_inp.weight`).
2. In the imatrix callback, when `ask=true` for `GGML_OP_MUL_MAT` with this tensor name, opt in to collection.
3. Save the full logit vector (128 dims) for each token at each layer.
4. Offline: for each layer, compare which experts are selected under top_k=8 vs top_k=4, compute overlap statistics.

**Pros**:
- Lightweight — only captures logits, not full hidden states
- Can compute routing overlap, score margin (gap between 4th and 5th expert), etc.

**Cons**:
- Only captures routing decisions, not actual output differences
- Cannot measure how much the MoE output changes — only whether different experts are selected
- Score margin might be a useful proxy but needs validation

---

## Comparison

| Criterion | Approach A (offline replay) | Approach B (per-layer inference) | Approach C (router logits) |
|-----------|---------------------------|--------------------------------|---------------------------|
| Accuracy | High (real inputs, isolated) | Highest (end-to-end) | Low (proxy only) |
| Speed | Fast (one inference + offline) | Slow (48 full runs) | Fast (one inference) |
| Implementation effort | Medium | Medium-High | Low |
| Captures cascading effects | No | Yes | No |
| Distribution shift free | Yes | Yes | Yes |
| Disk/memory cost | High (hidden states) | Low | Low |

---

## Feasibility Analysis

### Model Dimensions (Qwen3 MoE)

| Parameter | Value |
|-----------|-------|
| `n_embd` | 2048 |
| `n_ff` (per expert) | 768 |
| `n_expert` | 128 |
| `n_expert_used` | 8 |
| MoE layers | 48 (all transformer blocks) |
| Calibration tokens | ~430,080 |

### MoE Forward Pass (from `src/llama-graph.cpp` `build_moe_ffn()`)

```
hidden_state [n_embd, n_tokens]
  → MUL_MAT with ffn_gate_inp [n_embd, n_expert]     → router logits [n_expert, n_tokens]
  → softmax → argsort_top_k(8) → weight extraction + normalization (norm_w=true)
  → MUL_MAT_ID with ffn_gate_exps [n_ff, n_embd, n_expert]  (selected experts only)
  → MUL_MAT_ID with ffn_up_exps   [n_ff, n_embd, n_expert]
  → SwiGLU activation
  → MUL_MAT_ID with ffn_down_exps [n_embd, n_ff, n_expert]
  → weighted sum across 8 experts → MoE output [n_embd, n_tokens]
```

### Approach A Memory Estimates

Per-layer hidden state: 430K tokens × 2048 × 2 bytes (float16) = **1.68 GB**
All 48 layers: **80.6 GB** — too large to save at once.

Mitigation: sample 50K tokens → 48 × 50K × 2048 × 2 = **9.4 GB**, or process in chunks.

### Data Access in Callback

The imatrix eval callback receives the output tensor `t` at `ask=false` time. For a `GGML_OP_MUL_MAT` with `ffn_gate_inp`:
- `src0` = router weight `[n_embd, n_expert]`
- `src1` = hidden state `[n_embd, n_tokens]` (input activations)
- `t` = router logits `[n_expert, n_tokens]` (output, already computed)

The output tensor can be read from GPU using `ggml_backend_tensor_get()`, same pattern as `src1`. Tensor name pattern: `blk.{L}.ffn_gate_inp.weight`.

---

## Recommended Plan (Updated)

**Phase 1a** (implemented): Router-level analysis in callback.
- During inference, detect `ffn_gate_inp` tensors and read the output logits.
- Compute per-token softmax → sort → score margin (prob[3]-prob[4]) and weight concentration (top4/top8).
- Accumulate per-layer statistics; print summary table after inference.
- Cost: one additional GPU→CPU copy of 128×n_tokens floats per MoE layer (negligible).
- **Purpose**: determine whether router-level signals correlate with sensitivity. If layers with small score margin (routing is uncertain at the k=4 boundary) also show high perplexity impact, then this is a cheap and effective metric.

**Phase 1b** (if needed): Save hidden states for full offline replay.
- Modify callback to also save `src1` (hidden states before routing) to disk.
- Implement MoE forward pass in Python (numpy/torch) for offline replay with different top_k.
- Measure actual output difference (cosine similarity, L2, relative norm).
- Only pursue if Phase 1a router metrics are insufficient.

**Phase 2**: Per-layer inference validation (Approach B).
- Run perplexity experiments only for the most interesting layers identified by Phase 1.
- Validates whether router-level or offline-replay sensitivity correlates with actual perplexity impact.

---

## Phase 1a Implementation

**File**: `tools/imatrix-iccad/imatrix-iccad.cpp`

**Data structure** (`router_layer_stats`):
- `n_tokens`: total tokens processed
- k=4 boundary: `sum_score_margin`, `sum_score_margin_sq`, `sum_weight_conc`, `sum_weight_conc_sq`
- k=6 boundary: `sum_score_margin_k6`, `sum_score_margin_sq_k6`, `sum_weight_conc_k6`, `sum_weight_conc_sq_k6`
- Shared: `sum_max_prob`, `sum_entropy`

**Callback logic** (in the dense `GGML_OP_MUL_MAT` branch):
1. Check if `wname` contains `"ffn_gate_inp"`
2. Extract layer index via `process_tensor_name()`
3. Copy output tensor `t` from GPU (`ggml_backend_tensor_get`)
4. For each token: softmax over 128 logits → sort descending → compute metrics at k=4 and k=6 boundaries → accumulate

**Output**: summary table printed after inference with per-layer mean/std of all metrics at both boundaries. Also persisted to the imatrix GGUF file as a `imatrix.router_stats` tensor (shape `[12, n_layers]`, float32), so router analysis can be viewed later via `--show-statistics` without re-running inference. Backward compatible with v1 format (shape `[8, n_layers]`, k=4 only).

**Interpretation guide**:
- **Large score margin** → routing is confident at boundary → safe to reduce top_k
- **High weight concentration** → top-k experts carry most of the weight → safe to reduce
- **Low routing entropy** → routing is peaked → fewer experts needed
- **High max probability** → one expert dominates → safe to reduce

---

## Open Questions

1. **What distance metric best captures "sensitivity"?** Cosine similarity of outputs? L2 norm of difference? Relative norm change? Need to experiment (Phase 1b).
2. **How many tokens to sample for offline replay?** Saving all ~430K tokens per layer is ~1.68 GB/layer. A random 10K–50K subset may suffice.
3. ~~**Should we also test intermediate top_k values (e.g., 5, 6, 7)?**~~ Addressed: k=6 boundary metrics now collected alongside k=4 in a single pass.
4. **Does score margin correlate with perplexity impact?** This is the key validation question for Phase 1a. If yes, score margin is a cheap and reliable metric.
