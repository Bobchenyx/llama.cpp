# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Research Context (ICCAD Project)

This is the `iccad` branch of a personal fork of llama.cpp, used for ICCAD research. The goal is a paper combining two techniques applied to MoE models:

1. **Activation-based pruning**: selectively reduce `top_k` (number of activated experts) on a per-layer basis, using a layer-importance metric derived from activation statistics.
2. **Expert quantization**: apply different quantization levels to different layers' MoE expert weights (mixed-precision quantization), using a (potentially different) per-layer importance metric.

Both targets may use **different reference metrics** — e.g., one might use ZD Score while the other uses entropy or `Σ(Act²)` — but both are developed and modified from the **same base tool**: `tools/imatrix/`.

**Target models**: Qwen3 MoE and Qwen3.5 MoE series.

**Primary focus area**: `tools/imatrix-iccad/` — our working copy of the imatrix tool where all ICCAD modifications are made. `tools/imatrix/` is kept as a fixed bug-fixed upstream baseline for reference and diffing.

### imatrix Workflow

```bash
# Build (CUDA recommended for speed)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release --target llama-imatrix llama-imatrix-iccad llama-quantize llama-cli llama-server llama-perplexity

# Collect importance matrix from a calibration dataset
./build/bin/llama-imatrix -m model.gguf -f calibration-data.txt -o imatrix.gguf -ngl 99

# Inspect per-layer/per-tensor statistics
./build/bin/llama-imatrix --in-file imatrix.gguf --show-statistics

# Use imatrix to guide quantization
./build/bin/llama-quantize --imatrix imatrix.gguf model.gguf model-q4_k_m.gguf q4_k_m

# Benchmark quantized model quality (perplexity is the primary fast benchmark)
./build/bin/llama-perplexity -m model-q4_k_m.gguf -f calibration-data.txt -ngl 99
```

The imatrix tool collects **squared activation norms** for each tensor across calibration data chunks. Key statistics per tensor: `Σ(Act²)`, ZD Score (layer importance from arXiv:2406.17415), entropy, and cosine similarity to adjacent layers. These directly inform which layers to prune or quantize more aggressively.

The baseline source is `tools/imatrix/imatrix.cpp`. Our working file is `tools/imatrix-iccad/imatrix-iccad.cpp`.

## Documentation Maintenance

**All documentation must be written in English** (for team communication).

**Every session, keep all documents current:**

- After any meaningful discovery, code change, or experiment result, update the relevant file in `iccad/` immediately — do not defer to end of session.
- If a note or result is added to `iccad/`, update `iccad/README.md` index and the index in this file (`## Research Notes & Results`) in the same commit.
- If CLAUDE.md itself becomes outdated (e.g., a build flag changes, a new tool is added), update it before moving on.
- Commit message convention for documentation: `iccad : <short description>`.

## AI Usage Policy

This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be used only in an assistive capacity — corrections, expanding on verbose modifications already conceived by a human contributor, etc. All AI usage requires explicit disclosure. See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for the full policy.

## Build Commands

This project targets **CPU and CUDA only**. Other backends (Metal, Vulkan, HIP, SYCL, etc.) are out of scope.

```bash
# Standard build
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release

# Debug build
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Build only the relevant targets (faster iteration)
cmake --build build --config Release --target llama-imatrix llama-imatrix-iccad llama-quantize llama-cli llama-server llama-perplexity
```

## Testing

Tests use CTest. After building:

```bash
# Run all tests
cd build && ctest --output-on-failure

# Run a single test by name
cd build && ctest -R test-tokenizer-0 --output-on-failure

# Run tests with a label
cd build && ctest -L main --output-on-failure

# Run the full CI locally (CPU)
bash ./ci/run.sh ./tmp/results ./tmp/mnt

# With CUDA
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

```

Test binaries are built to `build/bin/`. Test sources live in `tests/`.

After modifying `ggml` source or operators, run `test-backend-ops` to verify consistency across backends.

## Architecture

### Layer Overview

```
ggml/          - Low-level tensor library (the compute engine)
src/           - llama library implementation (internal headers/sources)
include/       - Public C API (llama.h, llama-cpp.h)
common/        - Shared utilities for tools and examples
tools/         - End-user binaries (server, cli, quantize, bench, etc.)
examples/      - Minimal usage examples
tests/         - Unit and integration tests
```

### Key Layers

**ggml** (`ggml/include/ggml.h`, `ggml/src/`): The tensor computation library. Defines `ggml_tensor`, `ggml_context`, and the graph execution engine. Relevant backend implementations are `ggml/src/ggml-cpu/` and `ggml/src/ggml-cuda/`. The backend API is in `ggml/include/ggml-backend.h`.

**libllama** (`include/llama.h`, `src/`): The main public C library. Internal implementation is split into focused files:
- `src/llama-model.h/cpp` — model weight loading and architecture
- `src/llama-context.h/cpp` — inference context, batching, and decoding
- `src/llama-kv-cache.h/cpp` — KV cache management
- `src/llama-graph.h/cpp` — compute graph construction
- `src/llama-vocab.h/cpp` — tokenizer/vocabulary
- `src/llama-arch.h/cpp` — architecture-specific parameters
- `src/llama-sampler.h/cpp` — sampling logic
- `src/llama-model-loader.h/cpp` — GGUF file I/O

**common** (`common/`): Shared C++ utilities used by tools and examples. Includes argument parsing (`arg.h`), chat template handling (`chat.h`), sampling wrappers (`sampling.h`), JSON schema to grammar conversion, and HTTP utilities.

**llama-server** (`tools/server/`): OpenAI-compatible HTTP server. Single-threaded `server_context` processes all inference; HTTP workers communicate via `server_queue`/`server_response` thread-safe queues. Each parallel inference slot is a `server_slot`. See `tools/server/README-dev.md` for detailed internals.

**llama-cli** (`tools/cli/`): CLI for interactive inference and experimentation.

### GGUF Format

Models are stored in GGUF format. Conversion scripts (`convert_*.py`) transform other formats to GGUF. The GGUF reader/writer is in `src/llama-model-loader.h` and `ggml/include/gguf.h`.

### Matrix Multiplication Convention

`C = ggml_mul_mat(ctx, A, B)` computes $C = B A^T$ (i.e., $C^T = A B^T$). This is the opposite of the standard convention.

## Coding Guidelines

- 4-space indentation, brackets on same line, `void * ptr`, `int & a`
- `snake_case` for all function, variable, and type names
- Enum values: `UPPER_CASE` prefixed with the enum name (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- Naming pattern: `<class>_<action>_<noun>` — e.g., `llama_model_init`, `llama_sampler_chain_remove`
- Use `init`/`free` for constructors/destructors
- Use sized integer types (`int32_t`, `uint8_t`) in public APIs
- Declare structs as `struct foo {}` not `typedef struct foo {} foo`
- No fancy STL — prefer simple `for` loops, avoid templates where possible
- Vertical alignment for readability in related declarations
- C/C++ filenames: all lowercase with dashes (`.h`/`.cpp`/`.c`); Python: lowercase with underscores
- Use `clang-format` (clang-tools v15+) when in doubt about formatting
- Avoid adding third-party dependencies or extra headers
- New model support: start with CPU-only in the initial PR

## Research Notes & Results

All research documents live in `iccad/`. See [iccad/README.md](iccad/README.md) for the full index.

### Notes
| File | Content |
|------|---------|
| [imatrix-code-analysis.md](iccad/notes/imatrix-code-analysis.md) | Complete walkthrough of `tools/imatrix/imatrix.cpp`: data flow, collection logic for dense vs MoE layers, statistics, file formats, and extension points for our research |
| [expert-activation-analysis.md](iccad/notes/expert-activation-analysis.md) | Analysis of per-expert activation distributions for Qwen3 MoE (E8 vs E4). Candidate metrics for top_k selection: Gini, ΔEntropy, ffn_down Σ(Act²) ratio. Preliminary layer groupings for mixed top_k schedule. |
| [per-layer-sensitivity-design.md](iccad/notes/per-layer-sensitivity-design.md) | Design doc: per-layer top_k sensitivity analysis via offline replay (save MLP inputs, replay with different top_k). Addresses distribution shift in pure E8/E4 comparison. |

### Results
| Description | Date |
|-------------|------|
| Qwen3 MoE imatrix E8 vs E4 expert activation distributions collected and analyzed | 2026-03-17 |
| ffn_down Σ(Act²) ratio (E4/E8) grows with depth but is magnitude scaling only (reference, not selection criterion) | 2026-03-17 |
| Preliminary candidate metrics: Gini, ΔEntropy, ffn_down ratio (reference only) | 2026-03-17 |

## Adding a New Model

See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md).

## Important Documentation

- [Build guide](docs/build.md)
- [Server dev docs](tools/server/README-dev.md)
- [GBNF grammars](grammars/README.md)
- [Quantization](tools/quantize/README.md)
- [Perplexity tool](tools/perplexity/README.md)
