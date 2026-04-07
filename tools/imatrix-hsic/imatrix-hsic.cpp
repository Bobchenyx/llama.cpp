// imatrix-hsic: Collect per-layer hidden representations for CKA/HSIC analysis.
//
// Captures model input embedding and per-layer ffn_moe_out tensors during
// calibration inference, averages each across tokens per chunk, and writes
// raw float32 binary files for offline CKA computation.
//
// Part of the ICCAD research project (iccad branch).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <numeric>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// ------------------------------------------------------------------------------------------------
// HSICCollector: accumulates per-chunk mean hidden representations
// ------------------------------------------------------------------------------------------------

struct HSICCollector {
    int32_t n_layers = 0;
    int32_t n_embd   = 0;
    int32_t n_ctx    = 512;
    std::string out_prefix = "hsic";

    // Completed chunk means: [chunk_idx][embd_dim]
    std::vector<std::vector<float>> embed_means;
    // [layer_idx][chunk_idx][embd_dim]
    std::vector<std::vector<std::vector<float>>> layer_means;

    // Current-chunk accumulators
    std::vector<float> embed_accum;                // [n_embd]
    std::vector<std::vector<float>> layer_accum;   // [n_layers][n_embd]
    int32_t embed_count = 0;
    std::vector<int32_t> layer_count;              // [n_layers]

    // Reusable GPU readback buffer
    std::vector<char> gpu_buf;

    std::mutex mtx;

    void init(int32_t layers, int32_t embd, int32_t ctx, const std::string & prefix) {
        n_layers   = layers;
        n_embd     = embd;
        n_ctx      = ctx;
        out_prefix = prefix;

        embed_means.clear();
        layer_means.resize(n_layers);

        embed_accum.assign(n_embd, 0.0f);
        embed_count = 0;

        layer_accum.resize(n_layers);
        layer_count.assign(n_layers, 0);
        for (int i = 0; i < n_layers; ++i) {
            layer_accum[i].assign(n_embd, 0.0f);
        }
    }

    void finalize_chunk() {
        // Embedding
        if (embed_count > 0) {
            std::vector<float> mean(n_embd);
            const float inv = 1.0f / embed_count;
            for (int j = 0; j < n_embd; ++j) {
                mean[j] = embed_accum[j] * inv;
            }
            embed_means.push_back(std::move(mean));
        }
        std::fill(embed_accum.begin(), embed_accum.end(), 0.0f);
        embed_count = 0;

        // Layers
        for (int il = 0; il < n_layers; ++il) {
            if (layer_count[il] > 0) {
                std::vector<float> mean(n_embd);
                const float inv = 1.0f / layer_count[il];
                for (int j = 0; j < n_embd; ++j) {
                    mean[j] = layer_accum[il][j] * inv;
                }
                layer_means[il].push_back(std::move(mean));
            }
            std::fill(layer_accum[il].begin(), layer_accum[il].end(), 0.0f);
            layer_count[il] = 0;
        }
    }

    void save() const {
        const int n_chunks = (int)embed_means.size();
        if (n_chunks == 0) {
            LOG_ERR("No chunks collected, nothing to save.\n");
            return;
        }

        LOG_INF("Saving HSIC probe data: %d chunks, %d layers, n_embd=%d\n", n_chunks, n_layers, n_embd);

        // Helper to write a [n_chunks, n_embd] matrix to a binary file
        auto write_bin = [&](const std::string & path, const std::vector<std::vector<float>> & data) {
            FILE * fp = fopen(path.c_str(), "wb");
            if (!fp) {
                LOG_ERR("Failed to open %s for writing\n", path.c_str());
                return;
            }
            for (const auto & row : data) {
                fwrite(row.data(), sizeof(float), n_embd, fp);
            }
            fclose(fp);
            LOG_INF("  wrote %s [%d, %d]\n", path.c_str(), (int)data.size(), n_embd);
        };

        // Input embedding
        write_bin(out_prefix + "_input.bin", embed_means);

        // Per-layer outputs
        for (int il = 0; il < n_layers; ++il) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s_layer_%02d.bin", out_prefix.c_str(), il);
            write_bin(fname, layer_means[il]);
        }

        // Last layer (separate copy)
        if (!layer_means.empty() && !layer_means[n_layers - 1].empty()) {
            write_bin(out_prefix + "_last_layer.bin", layer_means[n_layers - 1]);
        }

        // Metadata JSON
        {
            std::string meta_path = out_prefix + "_meta.json";
            FILE * fp = fopen(meta_path.c_str(), "w");
            if (!fp) {
                LOG_ERR("Failed to open %s for writing\n", meta_path.c_str());
                return;
            }
            fprintf(fp, "{\n");
            fprintf(fp, "  \"n_layers\": %d,\n", n_layers);
            fprintf(fp, "  \"n_chunks\": %d,\n", n_chunks);
            fprintf(fp, "  \"n_embd\": %d,\n", n_embd);
            fprintf(fp, "  \"n_ctx\": %d,\n", n_ctx);
            fprintf(fp, "  \"dtype\": \"float32\",\n");
            fprintf(fp, "  \"input_file\": \"%s_input.bin\",\n", out_prefix.c_str());
            fprintf(fp, "  \"last_layer_file\": \"%s_last_layer.bin\",\n", out_prefix.c_str());
            fprintf(fp, "  \"layer_files\": [");
            for (int il = 0; il < n_layers; ++il) {
                if (il > 0) fprintf(fp, ", ");
                fprintf(fp, "\"%s_layer_%02d.bin\"", out_prefix.c_str(), il);
            }
            fprintf(fp, "]\n");
            fprintf(fp, "}\n");
            fclose(fp);
            LOG_INF("  wrote %s\n", meta_path.c_str());
        }
    }
};

static HSICCollector g_collector;

// ------------------------------------------------------------------------------------------------
// StreamingCKA: accumulate covariance matrices for exact token-level CKA
// ------------------------------------------------------------------------------------------------

struct StreamingCKA {
    int n_embd   = 0;
    int n_layers = 0;
    int n_ctx    = 0;
    int64_t n_tokens_total = 0;

    // Per-chunk token buffers (float, cleared each chunk)
    std::vector<float> embed_buf;                 // [n_ctx * n_embd]
    std::vector<std::vector<float>> layer_buf;    // [n_layers][n_ctx * n_embd]
    int chunk_tokens = 0;

    // Accumulated statistics (double precision)
    std::vector<double> sum_embed;                // [n_embd]
    std::vector<double> gram_embed;               // [n_embd * n_embd]

    struct LayerStats {
        std::vector<double> sum;                  // [n_embd]
        std::vector<double> gram_self;            // [n_embd * n_embd]
        std::vector<double> gram_input;           // [n_embd * n_embd]
        std::vector<double> gram_last;            // [n_embd * n_embd]
    };
    std::vector<LayerStats> layers;

    // Reusable GPU readback buffer
    std::vector<char> gpu_buf;
    std::mutex mtx;

    void init(int embd, int nl, int ctx) {
        n_embd   = embd;
        n_layers = nl;
        n_ctx    = ctx;

        const size_t pe = (size_t)n_embd;
        const size_t pp = pe * pe;

        embed_buf.assign(n_ctx * pe, 0.0f);
        layer_buf.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            layer_buf[i].assign(n_ctx * pe, 0.0f);
        }
        chunk_tokens = 0;

        sum_embed.assign(pe, 0.0);
        gram_embed.assign(pp, 0.0);

        layers.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            layers[i].sum.assign(pe, 0.0);
            layers[i].gram_self.assign(pp, 0.0);
            layers[i].gram_input.assign(pp, 0.0);
            layers[i].gram_last.assign(pp, 0.0);
        }

        n_tokens_total = 0;
    }

    // Compute G += A^T @ B where A,B are [nt, p] float matrices, G is [p, p] double.
    // Uses float intermediate with tiling for cache efficiency, then accumulates to double.
    void accumulate_gram(std::vector<double> & G,
                         const float * A, const float * B, int nt) {
        const int p = n_embd;

        // Phase 1: compute C = A^T @ B in float with cache-friendly tiling
        thread_local std::vector<float> C;
        C.assign((size_t)p * p, 0.0f);

        constexpr int TILE = 64;
        for (int j0 = 0; j0 < p; j0 += TILE) {
            const int j1 = std::min(j0 + TILE, p);
            for (int k0 = 0; k0 < p; k0 += TILE) {
                const int k1 = std::min(k0 + TILE, p);
                const int kw = k1 - k0;
                for (int t = 0; t < nt; ++t) {
                    const float * a = A + t * p;
                    const float * b = B + t * p + k0;
                    for (int j = j0; j < j1; ++j) {
                        const float aj = a[j];
                        float * crow = C.data() + j * p + k0;
                        for (int k = 0; k < kw; ++k) {
                            crow[k] += aj * b[k];
                        }
                    }
                }
            }
        }

        // Phase 2: accumulate float result into double
        for (size_t i = 0; i < (size_t)p * p; ++i) {
            G[i] += (double)C[i];
        }
    }

    void accumulate_sum(std::vector<double> & S, const float * A, int nt) {
        const int p = n_embd;
        for (int t = 0; t < nt; ++t) {
            const float * a = A + t * p;
            for (int j = 0; j < p; ++j) {
                S[j] += (double)a[j];
            }
        }
    }

    void finalize_chunk() {
        const int nt = chunk_tokens;
        if (nt == 0) return;

        const float * E = embed_buf.data();
        const int last = n_layers - 1;
        const float * L = layer_buf[last].data();

        // Embed self-gram and sum (single-threaded, runs once)
        accumulate_gram(gram_embed, E, E, nt);
        accumulate_sum(sum_embed, E, nt);

        // Per-layer grams (parallelized — each layer is independent)
        #pragma omp parallel for schedule(dynamic)
        for (int il = 0; il < n_layers; ++il) {
            const float * H = layer_buf[il].data();
            accumulate_sum(layers[il].sum, H, nt);
            accumulate_gram(layers[il].gram_self,  H, H, nt);
            accumulate_gram(layers[il].gram_input,  E, H, nt);
            accumulate_gram(layers[il].gram_last,   L, H, nt);
        }

        n_tokens_total += nt;

        // Clear buffers
        chunk_tokens = 0;
        std::fill(embed_buf.begin(), embed_buf.end(), 0.0f);
        for (int il = 0; il < n_layers; ++il) {
            std::fill(layer_buf[il].begin(), layer_buf[il].end(), 0.0f);
        }
    }

    // Compute ||C||_F^2 where C = G - (1/n) * sa * sb^T, and G/sa/sb are accumulated
    double centered_fro_sq(const std::vector<double> & G,
                           const std::vector<double> & sa,
                           const std::vector<double> & sb) const {
        const int p = n_embd;
        const double inv_n = 1.0 / (double)n_tokens_total;
        double fro = 0.0;
        for (int j = 0; j < p; ++j) {
            for (int k = 0; k < p; ++k) {
                double c = G[j * p + k] - inv_n * sa[j] * sb[k];
                fro += c * c;
            }
        }
        return fro;
    }

    double compute_cka(const std::vector<double> & G_AB,
                       const std::vector<double> & G_AA,
                       const std::vector<double> & G_BB,
                       const std::vector<double> & sA,
                       const std::vector<double> & sB) const {
        double fro_ab = centered_fro_sq(G_AB, sA, sB);
        double fro_aa = centered_fro_sq(G_AA, sA, sA);
        double fro_bb = centered_fro_sq(G_BB, sB, sB);
        if (fro_aa <= 0.0 || fro_bb <= 0.0) return 0.0;
        return fro_ab / std::sqrt(fro_aa * fro_bb);
    }

    void print_results() const {
        LOG_INF("\n");
        LOG_INF("Streaming CKA Results (token-level, %lld tokens)\n", (long long)n_tokens_total);
        LOG_INF("==========================================================================================\n");
        LOG_INF("  L |   S_in  |  S_out  |     M1     |     M2     |     M3    \n");
        LOG_INF("    |         |         | Sout(1-Sin) | Sout/Sin   | Sout-Sin  \n");
        LOG_INF("------------------------------------------------------------------------------------------\n");

        const int last = n_layers - 1;
        std::vector<double> s_in(n_layers), s_out(n_layers), m1(n_layers);

        for (int il = 0; il < n_layers; ++il) {
            s_in[il]  = compute_cka(layers[il].gram_input, gram_embed,
                                    layers[il].gram_self, sum_embed, layers[il].sum);
            s_out[il] = compute_cka(layers[il].gram_last, layers[last].gram_self,
                                    layers[il].gram_self, layers[last].sum, layers[il].sum);
            m1[il] = s_out[il] * (1.0 - s_in[il]);
            double m2 = (s_in[il] > 0.0) ? s_out[il] / s_in[il] : 0.0;
            double m3 = s_out[il] - s_in[il];

            if (il == last) {
                LOG_INF("%3d | %7.4f | %7.4f | %7.4f   * | %7.4f   * | %7.4f   *\n",
                        il, s_in[il], s_out[il], m1[il], m2, m3);
            } else {
                LOG_INF("%3d | %7.4f | %7.4f | %7.4f     | %7.4f     | %7.4f    \n",
                        il, s_in[il], s_out[il], m1[il], m2, m3);
            }
        }

        // M1 ranking (exclude last layer)
        std::vector<int> rank_layers;
        for (int i = 0; i < n_layers; ++i) {
            if (i != last) rank_layers.push_back(i);
        }
        std::vector<int> order(rank_layers.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return m1[rank_layers[a]] > m1[rank_layers[b]];
        });

        int n_show = std::min(12, (int)rank_layers.size());
        LOG_INF("\nM1 top-%d:    [", n_show);
        for (int i = 0; i < n_show; ++i) {
            if (i > 0) LOG_INF(", ");
            LOG_INF("%d", rank_layers[order[i]]);
        }
        LOG_INF("]\n");
        LOG_INF("M1 bottom-%d: [", n_show);
        for (int i = 0; i < n_show; ++i) {
            if (i > 0) LOG_INF(", ");
            LOG_INF("%d", rank_layers[order[rank_layers.size() - 1 - i]]);
        }
        LOG_INF("]\n");
    }
};

static StreamingCKA g_streaming;
static bool g_use_streaming = false;

// ------------------------------------------------------------------------------------------------
// Eval callback: intercept named tensors during inference
// ------------------------------------------------------------------------------------------------

static bool hsic_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    GGML_UNUSED(user_data);

    const char * name = t->name;

    if (ask) {
        if (strcmp(name, "embd") == 0)              return true;
        if (strcmp(name, "model.input_embed") == 0)  return true;
        if (strncmp(name, "ffn_moe_out-", 12) == 0) return true;
        return false;
    }

    // Read tensor data (copy from GPU if needed)
    const float * ptr;
    std::vector<char> & gbuf = g_use_streaming ? g_streaming.gpu_buf : g_collector.gpu_buf;
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    if (!is_host) {
        const size_t nbytes = ggml_nbytes(t);
        gbuf.resize(nbytes);
        ggml_backend_tensor_get(t, gbuf.data(), 0, nbytes);
        ptr = (const float *)gbuf.data();
    } else {
        ptr = (const float *)t->data;
    }

    const int64_t n_embd   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    if (g_use_streaming) {
        std::lock_guard<std::mutex> lock(g_streaming.mtx);

        if (strcmp(name, "embd") == 0 || strcmp(name, "model.input_embed") == 0) {
            // Embed fires first — use chunk_tokens as offset, then advance
            const int base = g_streaming.chunk_tokens;
            memcpy(g_streaming.embed_buf.data() + base * n_embd,
                   ptr, n_tokens * n_embd * sizeof(float));
            g_streaming.chunk_tokens = base + (int)n_tokens;
        }
        else if (strncmp(name, "ffn_moe_out-", 12) == 0) {
            const int il = atoi(name + 12);
            if (il >= 0 && il < g_streaming.n_layers) {
                // Layer callbacks write at offset 0 for this batch
                // (embed already advanced chunk_tokens; layers see the same tokens)
                const int base = g_streaming.chunk_tokens - (int)n_tokens;
                memcpy(g_streaming.layer_buf[il].data() + base * n_embd,
                       ptr, n_tokens * n_embd * sizeof(float));
            }
        }
    } else {
        std::lock_guard<std::mutex> lock(g_collector.mtx);

        if (strcmp(name, "embd") == 0 || strcmp(name, "model.input_embed") == 0) {
            for (int64_t tok = 0; tok < n_tokens; ++tok) {
                const float * row = ptr + tok * n_embd;
                for (int64_t j = 0; j < n_embd; ++j) {
                    g_collector.embed_accum[j] += row[j];
                }
            }
            g_collector.embed_count += (int32_t)n_tokens;
        }
        else if (strncmp(name, "ffn_moe_out-", 12) == 0) {
            const int layer_idx = atoi(name + 12);
            if (layer_idx >= 0 && layer_idx < g_collector.n_layers) {
                for (int64_t tok = 0; tok < n_tokens; ++tok) {
                    const float * row = ptr + tok * n_embd;
                    for (int64_t j = 0; j < n_embd; ++j) {
                        g_collector.layer_accum[layer_idx][j] += row[j];
                    }
                }
                g_collector.layer_count[layer_idx] += (int32_t)n_tokens;
            }
        }
    }

    return true;
}

// ------------------------------------------------------------------------------------------------
// Inference loop: simplified from compute_imatrix()
// ------------------------------------------------------------------------------------------------

static bool compute_hsic(llama_context * ctx, const common_params & params, const int32_t n_ctx) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    auto tim1 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenizing the input ..\n", __func__);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, params.parse_special);

    auto tim2 = std::chrono::high_resolution_clock::now();
    LOG_INF("%s: tokenization took %g ms\n", __func__,
            1e-3 * std::chrono::duration_cast<std::chrono::microseconds>(tim2 - tim1).count());

    if (params.i_chunk > 0) {
        if (size_t((params.i_chunk + 2) * n_ctx) >= tokens.size()) {
            LOG_ERR("%s: not enough tokens after removing %d chunks\n", __func__, params.i_chunk);
            return false;
        }
        tokens.erase(tokens.begin(), tokens.begin() + params.i_chunk * n_ctx);
    }

    if ((int)tokens.size() < 2 * n_ctx) {
        LOG_ERR("%s: need at least %d tokens (got %zu)\n", __func__, 2 * n_ctx, tokens.size());
        return false;
    }

    const int n_chunk_max = tokens.size() / n_ctx;
    const int n_chunk = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);

    // Force single-sequence, one chunk per decode
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    LOG_INF("%s: collecting over %d chunks, n_ctx=%d\n", __func__, n_chunk, n_ctx);

    for (int i = 0; i < n_chunk; ++i) {
        const int start = i * n_ctx;
        const auto t_start = std::chrono::high_resolution_clock::now();

        llama_memory_clear(llama_get_memory(ctx), true);

        common_batch_clear(batch);

        const auto token_org = tokens[start];
        if (add_bos) {
            tokens[start] = llama_vocab_bos(vocab);
        }
        for (int k = 0; k < n_ctx; ++k) {
            common_batch_add(batch, tokens[start + k], k, {0}, true);
        }
        tokens[start] = token_org;

        if (llama_decode(ctx, batch)) {
            LOG_ERR("%s: failed to decode chunk %d\n", __func__, i);
            llama_batch_free(batch);
            return false;
        }

        if (g_use_streaming) {
            g_streaming.finalize_chunk();
        } else {
            g_collector.finalize_chunk();
        }

        if (i == 0) {
            llama_synchronize(ctx);
            const auto t_end = std::chrono::high_resolution_clock::now();
            const float t_total = std::chrono::duration<float>(t_end - t_start).count();
            LOG_INF("%s: %.2f seconds per pass - ETA ", __func__, t_total);
            int total_seconds = (int)(t_total * n_chunk);
            if (total_seconds >= 60 * 60) {
                LOG("%d hours ", total_seconds / (60 * 60));
                total_seconds = total_seconds % (60 * 60);
            }
            LOG("%.2f minutes\n", total_seconds / 60.0);
        }

        if ((i + 1) % 50 == 0 || i == n_chunk - 1) {
            LOG_INF("[%d/%d chunks]\n", i + 1, n_chunk);
        }
    }

    llama_batch_free(batch);
    return true;
}

// ------------------------------------------------------------------------------------------------
// Main
// ------------------------------------------------------------------------------------------------

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -f calibration.txt -o hsic_output -ngl 99\n\n", argv[0]);
    LOG("Captures per-layer MoE FFN outputs averaged per chunk for offline CKA/HSIC analysis.\n");
    LOG("Output: {prefix}_input.bin, {prefix}_layer_XX.bin, {prefix}_last_layer.bin, {prefix}_meta.json\n\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file = "hsic";
    params.n_ctx    = 512;
    params.escape   = false;

    // Check for --streaming-cka before common_params_parse (which doesn't know it)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--streaming-cka") == 0) {
            g_use_streaming = true;
            // Remove from argv so common_params_parse doesn't complain
            for (int j = i; j < argc - 1; ++j) argv[j] = argv[j + 1];
            --argc;
            --i;
        }
    }

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX, print_usage)) {
        return 1;
    }

    if (params.prompt.empty()) {
        LOG_ERR("Error: calibration data required (-f <file>)\n");
        return 1;
    }

    common_init();

    const int32_t n_ctx = params.n_ctx;
    if (n_ctx <= 0) {
        LOG_ERR("%s: requires --ctx-size > 0\n", __func__);
        return 1;
    }

    // Force single-sequence processing
    params.n_parallel = 1;
    params.n_batch    = n_ctx;
    params.n_ctx      = n_ctx;
    params.compute_ppl = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = hsic_callback;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const int32_t n_layers = llama_model_n_layer(model);
    const int32_t n_embd   = llama_model_n_embd(model);

    LOG_INF("%s: model has %d layers, n_embd=%d\n", __func__, n_layers, n_embd);

    if (g_use_streaming) {
        LOG_INF("%s: streaming CKA mode — accumulating token-level covariance matrices\n", __func__);
        g_streaming.init(n_embd, n_layers, n_ctx);
    } else {
        g_collector.init(n_layers, n_embd, n_ctx, params.out_file);
    }

    if (!compute_hsic(ctx, params, n_ctx)) {
        return 1;
    }

    if (g_use_streaming) {
        g_streaming.print_results();
    } else {
        g_collector.save();
    }

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();

    return 0;
}
