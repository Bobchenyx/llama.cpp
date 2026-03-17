# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.

## AI Usage Policy

This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be used only in an assistive capacity — corrections, expanding on verbose modifications already conceived by a human contributor, etc. All AI usage requires explicit disclosure. See [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for the full policy.

## Build Commands

```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release -j $(nproc)

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j $(nproc)

# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# Vulkan build
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j $(nproc)
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

# Run the full CI locally
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

**ggml** (`ggml/include/ggml.h`, `ggml/src/`): The tensor computation library. Defines `ggml_tensor`, `ggml_context`, and the graph execution engine. Backend implementations (CUDA, Metal, Vulkan, HIP, SYCL, etc.) each live in `ggml/src/ggml-<backend>/`. The backend API is in `ggml/include/ggml-backend.h`.

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

## Adding a New Model

See [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md).

## Important Documentation

- [Build guide](docs/build.md)
- [Server dev docs](tools/server/README-dev.md)
- [GBNF grammars](grammars/README.md)
- [Quantization](tools/quantize/README.md)
- [Perplexity tool](tools/perplexity/README.md)
