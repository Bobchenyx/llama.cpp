# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## Research Context (ICCAD Project)

This is the `iccad` branch of a personal fork of llama.cpp, used for ICCAD research on MoE model optimization:

1. **Activation-based pruning**: per-layer `top_k` reduction guided by CKA layer importance (M1 metric).
2. **Expert quantization**: per-layer mixed-precision quantization for MoE expert weights.

Target models: Qwen3 MoE and Qwen3.5 MoE series.
Tools: `tools/imatrix-iccad/` (imatrix + router analysis), `tools/imatrix-hsic/` (CKA probe). Do NOT modify `tools/imatrix/` (upstream baseline).

## Documentation

All docs in English. Commit convention: `iccad : <description>`. Keep `iccad/` docs current after discoveries or code changes. See [iccad/README.md](iccad/README.md) for the research document index.

## AI Usage Policy

This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be used only in an assistive capacity. All AI usage requires explicit disclosure. See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Build Commands

This project targets **CPU and CUDA only**.

```bash
# Standard build
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF
cmake --build build --config Release

# Debug build
cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Build only the relevant targets (faster iteration)
cmake --build build --config Release --target llama-imatrix llama-imatrix-iccad llama-imatrix-hsic llama-quantize llama-cli llama-server llama-perplexity
```

## Testing

Tests use CTest. After building:

```bash
cd build && ctest --output-on-failure              # all tests
cd build && ctest -R test-tokenizer-0 --output-on-failure  # single test
cd build && ctest -L main --output-on-failure      # by label
bash ./ci/run.sh ./tmp/results ./tmp/mnt           # full CI (CPU)
GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt  # with CUDA
```

Test binaries: `build/bin/`. Sources: `tests/`.
After modifying `ggml` source, run `test-backend-ops` to verify backend consistency.

## Architecture

```
ggml/          - Low-level tensor library (compute engine)
src/           - llama library implementation
include/       - Public C API (llama.h, llama-cpp.h)
common/        - Shared utilities for tools and examples
tools/         - End-user binaries (server, cli, quantize, bench, etc.)
examples/      - Minimal usage examples
tests/         - Unit and integration tests
```

**ggml** (`ggml/include/ggml.h`, `ggml/src/`): Tensor computation library. Backends: `ggml-cpu/`, `ggml-cuda/`.

**libllama** (`include/llama.h`, `src/`): Key internal files:
- `llama-model.h/cpp` — weight loading and architecture
- `llama-context.h/cpp` — inference context, batching, decoding
- `llama-kv-cache.h/cpp` — KV cache management
- `llama-graph.h/cpp` — compute graph construction
- `llama-vocab.h/cpp` — tokenizer
- `llama-model-loader.h/cpp` — GGUF file I/O

**common** (`common/`): Argument parsing, chat templates, sampling wrappers, JSON schema→grammar.

**Matrix multiply convention**: `C = ggml_mul_mat(ctx, A, B)` computes C = BA^T.

## Coding Guidelines

- 4-space indentation, brackets on same line, `void * ptr`, `int & a`
- `snake_case` everywhere; enum values `UPPER_CASE` with enum prefix
- Naming: `<class>_<action>_<noun>` (e.g., `llama_model_init`)
- Use `init`/`free` for constructors/destructors
- Sized integer types (`int32_t`, `uint8_t`) in public APIs
- `struct foo {}` not `typedef struct foo {} foo`
- Simple `for` loops over fancy STL; avoid templates where possible
- C/C++ filenames: lowercase with dashes; Python: lowercase with underscores
- Use `clang-format` (v15+) when in doubt
- Avoid third-party dependencies

## References

- [Add a new model](docs/development/HOWTO-add-model.md)
- [Build guide](docs/build.md)
- [Server dev docs](tools/server/README-dev.md)
- [Quantization](tools/quantize/README.md)
- [Perplexity tool](tools/perplexity/README.md)
