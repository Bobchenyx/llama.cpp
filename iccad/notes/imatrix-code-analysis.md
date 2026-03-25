# imatrix Code Analysis

Baseline source: `tools/imatrix/imatrix.cpp` (fixed upstream, do not modify)
Working copy: `tools/imatrix-iccad/imatrix-iccad.cpp` (all ICCAD modifications go here)

---

## Bug Fixes Applied (2026-03-17)

Four divide-by-zero bugs fixed, all triggered when running `--show-statistics` on a MoE model imatrix.

| # | Location | Root Cause | Fix |
|---|----------|-----------|-----|
| 1 | `compute_statistics()` | `values[i] / counts[i]` when `counts[i] == 0` for unactivated MoE experts | Skip experts with zero count |
| 2 | `compute_statistics()` | `max_element` / `accumulate` on empty vector after all counts are zero | Return early if `activations` is empty |
| 3 | `compute_cossim()` | `dp / (curr_mag * prev_mag)` when tensor is all-zeros | Guard with `denom > 0.0f` check |
| 4 | `load_imatrix()` | `max_count / (m_params.n_ctx / m_params.n_parallel)` — integer division by zero because `g_collector.set_params()` is never called before `show_statistics()` in `main()`, leaving `m_params.n_ctx` at default value 0 | Read `m_last_chunk` directly from `imatrix.chunk_count` GGUF metadata key; fallback to division only if key is absent, with zero guard |

Fixes 1-2 sourced from upstream PRs: [ggml-org/llama.cpp#19532](https://github.com/ggml-org/llama.cpp/pull/19532) / [auroralabs-loci/llama.cpp#1219](https://github.com/auroralabs-loci/llama.cpp/pull/1219).
Fixes 3-4 identified and resolved independently.

---

## High-Level Flow

```
main()
 ├── Parse CLI arguments
 ├── (Optional) Load existing imatrix files (--in-file) and accumulate into memory
 ├── Register callback: params.cb_eval = ik_collect_imatrix
 ├── common_init_from_params() → initialize llama model + context
 └── compute_imatrix()
      ├── Tokenize calibration text
      ├── Split into chunks of size n_ctx, call llama_decode() per chunk
      │    └── During ggml graph execution, the callback collect_imatrix() fires for each node
      ├── (Optional) Compute perplexity (skip with --no-ppl)
      └── Call save_imatrix() when done
```

---

## Core Data Structures

```cpp
struct Stats {
    std::vector<float>   values;   // accumulated squared activations (see shapes below)
    std::vector<int64_t> counts;   // activation count — per-expert for MoE, or global for dense
};

// Global singleton
IMatrixCollector g_collector;
// Internal map: tensor name -> Stats
std::unordered_map<std::string, Stats> m_stats;
```

---

## Callback Mechanism: collect_imatrix()

During ggml graph execution, the callback is called **twice per node**:

1. **ask=true**: filter pass — return `true` to opt in, `false` to skip
2. **ask=false**: collection pass — actually accumulate data

### Filter Logic (ask=true)

```
GGML_OP_MUL_MAT_ID  → always collect (MoE sparse expert matmul)
GGML_OP_MUL_MAT     → collect only if:
    - src1 (activation) has >= 16 tokens  (small batches ignored)
    - src1 type is F32
    - tensor name starts with "blk."  (or output.weight if --process-output)
```

Tensor names are normalized by `filter_tensor_name()` to strip backend prefixes/suffixes:
```
CUDA0#blk.0.attn_k.weight#0  →  blk.0.attn_k.weight
```

---

## Data Collection Logic

### 1. Dense Layers (GGML_OP_MUL_MAT)

Matrix multiply: `output = src1 @ src0.T`, where `src1` is the input activation.

```
e.values  shape: [in_dim]    (src1.ne[0])
e.counts  shape: [1]

For each row x of src1 (one token's hidden state):
    e.values[j] += x[j] * x[j]    (element-wise squared accumulation)
e.counts[0] += n_rows_processed
```

### 2. MoE Sparse Layers (GGML_OP_MUL_MAT_ID)

```
src0: expert weight tensor,   shape [out_dim, in_dim, n_experts]
src1: input activations,      shape [in_dim, n_expert_used, n_tokens]
ids:  routing result (top-k), shape [n_expert_used, n_tokens]

e.values  shape: [in_dim * n_experts]   — independently accumulated per expert
e.counts  shape: [n_experts]            — independently counted per expert

For each expert ex in [0, n_experts):
    For each routing slot idx in [0, n_expert_used):
        For each token row:
            if ids[row][idx] == ex:        ← token was actually routed to this expert
                e.values[ex*in_dim + j] += x[j]^2
                e.counts[ex]++
```

**Key point**: each expert's activation statistics are tracked **independently**.
Experts never routed to have `counts=0`, and are flagged as partial data when saving.

---

## Normalization and Statistics (compute_statistics)

Raw stored values are cumulative. Normalized activation per element:

```
normalized_activation[i] = values[i] / counts[expert_of_i]
```

Statistics computed on top of this:

| Metric | Meaning |
|--------|---------|
| `Σ(Act²)` / `total_sqract` | Sum of all normalized activations — overall activation magnitude |
| `mean_sqract` / `μ` | Mean of normalized activations |
| `max_sqract` / `min_sqract` | Range extremes |
| `stddev` / `σ` | Standard deviation |
| `active` / `% Active` | Fraction of elements where avg x² > 1e-5 |
| `entropy` | Shannon entropy: $-\sum p_i \log_2 p_i$, where $p_i = act_i / \Sigma act$ |
| `zd` / ZD Score | Fraction of elements with z-score > 1 (from [arXiv:2406.17415](https://arxiv.org/abs/2406.17415)) |
| `cossim` / CosSim | Cosine similarity of this layer's activation values vs. the same tensor type in the previous layer |

Per-layer aggregation: element-count-weighted average of `Σ(Act²)`, ZD, and CosSim across all tensors in each layer.

---

## File Formats

### GGUF (default, recommended)

Two ggml tensors stored per monitored tensor:
```
{name}.in_sum2   shape [in_dim, n_experts]   raw cumulative Σ(x²), NOT normalized
{name}.counts    shape [1, n_experts]         activation counts per expert
```
Storing raw (un-normalized) values allows **multiple imatrix files to be merged by simple addition** (`--in-file` accepts multiple inputs).

### Legacy .dat Format

Stores `(values[i] / counts) * ncall` — normalized then re-scaled. Lossy to merge. Not recommended for new work.

---

## Extension Points for Our Research

### 1. Callback Filter (control which ops are collected)

```cpp
// In collect_imatrix(), ask=true branch:
if (t->op == GGML_OP_MUL_MAT_ID) return true;
```

Can add filtering logic here — e.g., restrict collection to specific layer indices, or add new op types.

### 2. New Metrics (extend compute_statistics)

All current metrics are derived from `values[]` (squared input activations).
Potential additions:
- ~~Per-expert activation frequency: `counts[ex] / total_tokens`~~ **Implemented** — see ICCAD Extensions below
- Expert activation distribution uniformity (entropy over experts within a layer)

### 3. Output Format Extension (extend save_imatrix)

Can add a CSV/JSON output of per-expert stats for downstream analysis scripts without changing the GGUF format.

### 4. Interface with llama-quantize

`llama-quantize --imatrix imatrix.gguf` reads `in_sum2` and `counts` from the GGUF file, then uses `values[j] / counts` as per-feature importance weights during quantization — lower importance features are allowed larger quantization error.

Our per-layer mixed-precision quantization will require additional logic in `tools/quantize/` to read per-layer quantization configurations alongside the imatrix.

---

## Implications for Our Research

### Activation Pruning (top_k reduction)
- `counts[ex]` directly reflects how often each expert is activated across the calibration set
- `total_sqract` and `zd` reflect overall layer importance
- Layers with low `total_sqract` or high `cossim` (highly redundant with previous layer) → candidates for top_k reduction

### Expert Quantization (mixed precision)
- `e.values[ex*in_dim + j] / e.counts[ex]` = average squared activation of the j-th input feature for expert `ex`
- Aggregate this into a per-expert importance score → assign quantization precision accordingly
- Current quantize tool uses a merged imatrix across all experts; needs modification to support per-expert differentiation

---

## ICCAD Extensions (tools/imatrix-iccad/imatrix-iccad.cpp)

### Per-expert activation count display (2026-03-17)

Added to `--show-statistics` output. After the existing per-tensor and per-layer summary tables, a new section prints raw expert activation counts for every MoE transformer block.

**Data flow:**

1. `tensor_statistics` struct: added `std::vector<int64_t> expert_counts` field (non-empty only for MoE tensors where `n_mat > 1`).
2. `compute_statistics()`: at the end, if `n_mat > 1`, copies `e.counts` into `ts.expert_counts`.
3. `show_statistics()`: after the existing tables, collects one representative tensor per layer (all MoE tensors within a block share identical routing counts), then prints:

```
Per-expert activation counts — MoE layers (N layers, M experts each)
(Counts reflect total activations over the calibration dataset)

 Layer   Exp0    Exp1    ...    Total
------  ------  ------  ...  ------
     0   12345   11234  ...   98765
     1   ...
```

**Design decision**: No Top-K% columns. The experiment plan runs two separate model binaries (configured with `num_experts_per_token=8` and `num_experts_per_token=4`) and compares their imatrix outputs directly. The raw per-expert counts are sufficient for that comparison.

### Router-level analysis (2026-03-23, updated 2026-03-24)

Added to the inference callback (not `--show-statistics`). During imatrix collection, when a `GGML_OP_MUL_MAT` fires for a `ffn_gate_inp` (MoE router) tensor, the output tensor `t` (router logits `[n_expert, n_tokens]`) is read from GPU and analyzed per-token.

**Metrics computed per token at two boundaries (k=4 and k=6), accumulated per layer:**

| Metric | k=4 definition | k=6 definition | Sensitivity interpretation |
|--------|---------------|---------------|--------------------------|
| Score margin | `prob[3] - prob[4]` | `prob[5] - prob[6]` | Larger → routing is confident at boundary → safer to reduce top_k |
| Weight concentration | `top4_sum / top8_sum` | `top6_sum / top8_sum` | Higher → top-k experts carry most weight → safer to reduce |
| Max probability | max expert probability | (shared) | Higher → one expert dominates → fewer experts needed |
| Routing entropy | `-Σ p·log₂(p)` over all 128 experts | (shared) | Lower → more peaked routing → safer to reduce |

**Data flow:**

1. `router_layer_stats` struct: accumulates sums and squared sums for online mean/stddev computation. Tracks k=4 and k=6 boundary metrics separately; max_prob and entropy are shared.
2. `IMatrixCollector::m_router_stats`: `std::map<int, router_layer_stats>` keyed by layer index.
3. In `collect_imatrix()` dense MUL_MAT branch: detects `ffn_gate_inp` in tensor name, reads output `t` via `ggml_backend_tensor_get()`, computes softmax → sort → metrics at both boundaries for each token.
4. `print_router_stats()`: called after inference and in `--show-statistics` mode, prints per-layer summary table with k=4 and k=6 columns side-by-side.
5. `save_imatrix()`: writes accumulated router stats as a `imatrix.router_stats` tensor (shape `[12, n_layers]`, float32) in the GGUF file. Fields per row: layer_idx, n_tokens, SM_k4, SM_k4_sq, WC_k4, WC_k4_sq, max_prob, entropy, SM_k6, SM_k6_sq, WC_k6, WC_k6_sq.
6. `load_imatrix()`: reads the tensor back and merges into `m_router_stats` (additive). Backward compatible: accepts both v1 (ne[0]==8, k=4 only) and v2 (ne[0]==12, k=4+k=6) formats.

**Output format:**

```
Router analysis — per-layer score margin and weight concentration (ICCAD)
 Layer     Tokens      SM_k4  SM_k4_Std      WC_k4      SM_k6  SM_k6_Std      WC_k6    MaxProb  RoutEntropy
------  --------  ---------  ---------  ---------  ---------  ---------  ---------  ---------  -----------
     0    430080   0.007158   0.007033   0.696372   0.004486   0.004795   0.873375   0.087974       6.1587
```

---

## Implementation Notes

- **Small batch skip**: `src1->ne[1] < 16` is skipped to avoid prefill-phase noise
- **Unactivated MoE experts**: experts with `counts=0` trigger a partial data warning; quantize needs a fallback
- **GPU→CPU copy**: when activations are on GPU, `ggml_backend_tensor_get()` pulls them back to CPU — has non-trivial overhead at scale
- **Thread safety**: collection is mutex-protected (`m_mutex`); saving is single-threaded
