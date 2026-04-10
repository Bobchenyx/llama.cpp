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
| 4 | `load_imatrix()` | `max_count / (m_params.n_ctx / m_params.n_parallel)` — integer division by zero | Read `m_last_chunk` from `imatrix.chunk_count` GGUF key |

Fixes 1-2 from upstream PRs: [#19532](https://github.com/ggml-org/llama.cpp/pull/19532) / [auroralabs-loci#1219](https://github.com/auroralabs-loci/llama.cpp/pull/1219). Fixes 3-4 identified independently.

---

## High-Level Flow

```
main()
 ├── Parse CLI arguments
 ├── (Optional) Load existing imatrix files (--in-file)
 ├── Register callback: params.cb_eval = ik_collect_imatrix
 ├── common_init_from_params() → init model + context
 └── compute_imatrix()
      ├── Tokenize calibration text → chunks of n_ctx
      ├── llama_decode() per chunk → callback fires for each graph node
      └── save_imatrix() when done
```

## Core Data Structures

```cpp
struct Stats {
    std::vector<float>   values;   // accumulated Σ(x²) per element
    std::vector<int64_t> counts;   // per-expert for MoE, [1] for dense
};
// Global: std::unordered_map<std::string, Stats> m_stats;
```

## Callback: collect_imatrix()

Called twice per node: `ask=true` (filter), `ask=false` (collect).

**Filter**: `MUL_MAT_ID` always collected; `MUL_MAT` only if src1 has ≥16 tokens, F32, and name starts with `blk.`.

**Dense layers** (`MUL_MAT`): accumulate `x[j]²` into `values[j]`, increment `counts[0]`.

**MoE layers** (`MUL_MAT_ID`): each expert tracked independently via `ids[row][idx] == ex` routing check. `values` shape `[in_dim × n_experts]`, `counts` shape `[n_experts]`.

---

## ICCAD Extensions

### Per-expert activation count display (2026-03-17)

Added to `--show-statistics`. After existing tables, prints raw expert activation counts
per MoE layer (one representative tensor per block — all share routing).

### Router-level analysis (2026-03-23)

During inference, detects `ffn_gate_inp` tensors and reads router output logits from GPU.
Computes per-token: softmax → sort → score margin, weight concentration at k=4 and k=6
boundaries, max probability, routing entropy. Accumulated per-layer.

Persisted as `imatrix.router_stats` tensor in GGUF (v2: shape `[12, n_layers]`, float32).
Backward compatible with v1 format (`[8, n_layers]`, k=4 only).

See [results/router-analysis.md](../results/router-analysis.md) for findings.

---

## Implementation Notes

- **Small batch skip**: `src1->ne[1] < 16` skipped to avoid prefill-phase noise
- **Unactivated experts**: `counts=0` triggers partial data warning; quantize needs fallback
- **GPU→CPU copy**: `ggml_backend_tensor_get()` has non-trivial overhead at scale
- **Thread safety**: collection is mutex-protected; saving is single-threaded
