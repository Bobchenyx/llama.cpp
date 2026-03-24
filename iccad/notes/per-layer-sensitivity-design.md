# Per-Layer top_k Sensitivity Analysis — Design Document

**Date**: 2026-03-23
**Status**: Design phase

---

## Motivation

Our current analysis compares two "pure" configurations (all-E8 vs all-E4) and uses ΔEntropy to rank layer sensitivity. However, the actual deployment target is a **mixed top_k schedule** — some layers at top_k=8, others at top_k=4. This introduces a **distribution shift problem**: changing top_k at one layer alters the input distribution for all downstream layers.

Metrics derived from comparing pure E8 vs pure E4 may not accurately predict per-layer sensitivity in a mixed configuration. We need a method that isolates each layer's sensitivity independently.

---

## Goal

For each of the 48 MoE layers, measure **how much the layer's output changes** when its top_k is reduced (e.g., 8→4), while keeping all other layers at their baseline configuration (top_k=8). This gives a per-layer sensitivity score free from inter-layer distribution shift.

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

## Recommended Plan

**Phase 1**: Implement Approach A (offline replay) as the primary tool.
- Gives per-layer sensitivity scores quickly
- Independent of distribution shift
- Results can guide which layers to focus Approach B experiments on

**Phase 2**: Implement Approach B (per-layer inference) for validation.
- Run perplexity experiments only for the most interesting layers identified by Phase 1
- Validates whether offline replay sensitivity correlates with actual perplexity impact
- If correlation is strong, Approach A metrics can be trusted for future experiments

---

## Open Questions

1. **What distance metric best captures "sensitivity"?** Cosine similarity of outputs? L2 norm of difference? Relative norm change? Need to experiment.
2. **How many tokens to sample?** Saving all ~430K tokens per layer may be prohibitive. Is a random 10K–50K subset sufficient?
3. **Should we also test intermediate top_k values (e.g., 5, 6, 7)?** This would give a sensitivity curve per layer, not just a binary E8 vs E4 comparison.
4. **Router score margin**: the gap between the k-th and (k+1)-th expert score might be a cheap proxy for sensitivity — layers where the margin is small → more sensitive to top_k reduction. Worth computing alongside Approach C.
