# Per-Layer top_k Perplexity Validation

## Setup

- Model: Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf (48 MoE layers, default k=8)
- Benchmark: wikitext-2-raw (wiki.test.raw), ctx=512, 584 chunks
- Mechanism: per-layer `n_expert_used` schedule via `--override-kv expert_schedule=str:<path>`
- All schedule runs: 35 layers at k=4, 13 layers at k=8 (12 selected + layer 47 always kept)
- Layer 47 excluded from all schedules (last layer before output, always k=8)

## Results

| Run | Setting | 12 layers kept at k=8 | PPL | ΔPPL |
|-----|---------|----------------------|------|------|
| B1  | all k=8 (baseline) | all 48 | 7.3598 ± 0.052 | — |
| A1  | CKA M1 top-12 | [3,6,8,10,12,37,39,42,43,44,45,46] | 7.9905 ± 0.057 | +0.631 |
| A2  | CKA M2 top-12 | [0,8,10,11,12,37,39,42,43,44,45,46] | 7.9912 ± 0.057 | +0.631 |
| C   | uniform (il%4==0) | [0,4,8,12,16,20,24,28,32,36,40,44] | 8.1357 ± 0.058 | +0.776 |
| B   | CKA M1 bottom-12 | [0,5,9,19,21,26,28,31,32,33,34,38] | 8.2323 ± 0.059 | +0.873 |
| B2  | all k=4 | none | 8.6199 ± 0.062 | +1.260 |

## Ordering

```
B1 (all k=8)  <  A1 (M1 top-12)  ≈  A2 (M2 top-12)  <  C (uniform)  <  B (M1 bot-12)  <  B2 (all k=4)
   7.36            7.99                7.99                8.14            8.23               8.62
```

## Analysis

1. **CKA ranking validated**: M1 and M2 both identify better layers than uniform spacing or
   random (bottom-12) selection. PPL degradation is 50% of all-k4 (0.63 vs 1.26).

2. **M1 ≈ M2**: The two metrics produce nearly identical PPL (Δ=0.0007). This is expected given
   their high correlation (Spearman ρ=0.886). M1 is recommended as the primary metric since it
   avoids the ratio instability of M2 when S_in is small.

3. **Uniform baseline**: Uniform spacing (layer%4==0) is worse than CKA-guided selection by
   0.15 PPL points, showing that layer importance is NOT uniform across depth.

4. **Bottom-12 worst case**: Keeping the CKA-identified least important layers at k=8 gives
   only marginal improvement over all-k4 (8.23 vs 8.62), confirming these layers are indeed
   less sensitive to expert reduction.

5. **PPL recovery**: CKA-guided 12/48 selection recovers 50% of the PPL gap
   (1.26 total, 0.63 recovered). With 75% of layers pruned to k=4, only 8.6% PPL increase.

---

## Code Modifications

### Overview

llama.cpp uses a single global `n_expert_used` (number of activated experts per token) for
all MoE layers. To support per-layer top_k schedules, we need three changes:

1. **Storage**: per-layer array in `hparams`
2. **Loading**: schedule file parser in model loading
3. **Graph building**: per-layer value at the MoE call site

Plus a **bug fix** discovered during validation: the expert aggregation loop in `build_moe_ffn`
used the global value instead of the per-layer parameter, causing out-of-bounds memory reads.

### Change 1: Per-layer array (`src/llama-hparams.h:49`)

Add a per-layer expert count array alongside the existing global scalar:

```cpp
uint32_t n_expert_used = 0;
std::array<uint32_t, LLAMA_MAX_LAYERS> n_expert_used_arr = {};  // ICCAD: per-layer override
```

The global `n_expert_used` continues to serve as the model default and is used by the graph
context during warmup (`llama-graph.cpp:861`). The new array holds the actual per-layer values
used during inference.

### Change 2: Schedule loading (`src/llama-model.cpp:379-406`)

In `load_hparams()`, right after `n_expert_used` is read from the GGUF metadata (line 375),
two blocks are inserted:

**Block 1** — Initialize array from global default:

```cpp
for (uint32_t i = 0; i < hparams.n_layer; ++i) {
    hparams.n_expert_used_arr[i] = hparams.n_expert_used;
}
```

This runs BEFORE the per-architecture `switch` block, so it works for any MoE architecture.

**Block 2** — Load schedule file via `--override-kv`:

```cpp
std::string sched_path;
if (ml.get_key("expert_schedule", sched_path, false)) {
    FILE * fp = fopen(sched_path.c_str(), "r");
    // ... parse "layer_id top_k" lines, override n_expert_used_arr[layer] ...
}
```

The `ml.get_key("expert_schedule", ...)` checks KV overrides set via the command line. This
reuses the existing `--override-kv` mechanism without adding new CLI flags. Usage:

```bash
llama-perplexity -m model.gguf -f data.txt --override-kv expert_schedule=str:/path/to/schedule.txt
```

**Schedule file format**: one `layer_id top_k` pair per line. Unlisted layers keep the model
default. Example (`m1_keep_top12.txt` — 35 lines for layers reduced to k=4):

```
0 4
1 4
2 4
...
```

The position of this code (after line 375, before line 420's assertions) matters:
- It must be AFTER `n_expert_used` is read from GGUF (line 375)
- It must be BEFORE the `n_expert_used <= n_expert` assertion (line 421), which still checks
  the global value — the assertion validates the model metadata, not our schedule

### Change 3: Per-layer expert count in graph building (`src/models/qwen3moe.cpp:95`)

In the `build_moe_ffn()` call, replace the graph context member `n_expert_used` with the
per-layer array value:

```cpp
// Before:
n_expert, n_expert_used,

// After:
n_expert, cparams.warmup ? n_expert_used : (int64_t)hparams.n_expert_used_arr[il],
```

**Why the warmup guard**: During warmup (`cparams.warmup == true`), the graph context member
`n_expert_used` is set to `hparams.n_expert` (all 128 experts, see `llama-graph.cpp:861`).
This builds a "maximum size" graph so the backend allocates buffers large enough for any
configuration. We must preserve this behavior — using the per-layer value (e.g., 4) during
warmup would under-allocate buffers and crash during inference.

During normal inference (`cparams.warmup == false`), we use the per-layer value from the array.

### Change 4: Bug fix in expert aggregation (`src/llama-graph.cpp:1527`)

**The bug**: After `build_moe_ffn` computes the expert outputs into a tensor of shape
`[n_embd, n_expert_used, n_tokens]`, the expert outputs are split into views and summed.
The original code used `hparams.n_expert_used` (the global model default) for this loop:

```cpp
// ORIGINAL (buggy for per-layer schedule):
for (uint32_t i = 0; i < hparams.n_expert_used; ++i) {
    cur_experts[i] = ggml_view_2d(ctx0, experts, n_embd, n_tokens,
                                  experts->nb[2], i*experts->nb[1]);
}
```

This was intentional — the comment referenced PR #14753: during warmup, the function parameter
`n_expert_used` equals `hparams.n_expert` (128), and creating 128 add nodes is wasteful.
Using the global value (8) avoids this since the extra expert slots have near-zero weight.

**The problem**: With per-layer schedules, a layer might have `n_expert_used=4` (from the
schedule) while `hparams.n_expert_used=8` (model default). The expert tensor has shape
`[n_embd, 4, n_tokens]`, but the loop iterates `i=0..7`. Indices 4-7 read past the tensor
boundary, producing garbage values that corrupt the MoE output.

**Symptom**: PPL = 639-1649 (catastrophic) instead of expected ~8.

**The fix**: Use `std::min(n_expert_used, (int64_t)hparams.n_expert_used)`:

```cpp
const int64_t n_expert_agg = std::min(n_expert_used, (int64_t)hparams.n_expert_used);
for (int64_t i = 0; i < n_expert_agg; ++i) { ... }
```

This handles all three cases correctly:
- **Warmup** (n_expert_used=128, hparams=8): min=8, avoids excessive add nodes
- **Default inference** (n_expert_used=8, hparams=8): min=8, unchanged behavior
- **Per-layer k=4** (n_expert_used=4, hparams=8): min=4, correct aggregation

### Layer 47 constraint

Layer 47 (the final MoE layer, whose output feeds directly into `output_norm → lm_head`)
must ALWAYS keep the default top_k. Setting it to k=4 causes catastrophic PPL (766+).
This is consistent with CKA analysis: layer 47 was excluded from ranking because it serves
as the S_out reference (CKA(H_47, H_47) = 1.0 trivially), and its M1 score (0.6572) is
by far the highest — confirming it is the single most important layer.

All schedule files in `iccad/schedules/` exclude layer 47.

---

## Schedule Files

All schedules list the layers set to k=4 (unlisted layers keep default k=8).
Each file has 35 lines (35 layers at k=4 + 13 at k=8: 12 selected + layer 47).

```
iccad/schedules/
├── m1_keep_top12.txt       # A1: CKA M1 top-12 at k=8, other 35 at k=4
├── m2_keep_top12.txt       # A2: CKA M2 top-12 at k=8, other 35 at k=4
├── m1_keep_bot12.txt       # B:  CKA M1 bottom-12 at k=8, other 35 at k=4
└── uniform_keep_12.txt     # C:  layer%4==0 at k=8, other 35 at k=4
```

## Run Commands

```bash
MODEL=qwen3/Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf
DATA=wikitext-2-raw/wiki.test.raw
PPL=llama.cpp/build/bin/llama-perplexity
SCHED=llama.cpp/iccad/schedules

# Global baselines
$PPL -m $MODEL -f $DATA -ngl 99                                                    # B1: all k=8
$PPL -m $MODEL -f $DATA -ngl 99 --override-kv qwen3moe.expert_used_count=int:4     # B2: all k=4

# Per-layer schedule runs
$PPL -m $MODEL -f $DATA -ngl 99 --override-kv expert_schedule=str:$SCHED/m1_keep_top12.txt    # A1
$PPL -m $MODEL -f $DATA -ngl 99 --override-kv expert_schedule=str:$SCHED/m2_keep_top12.txt    # A2
$PPL -m $MODEL -f $DATA -ngl 99 --override-kv expert_schedule=str:$SCHED/m1_keep_bot12.txt    # B
$PPL -m $MODEL -f $DATA -ngl 99 --override-kv expert_schedule=str:$SCHED/uniform_keep_12.txt  # C
```
