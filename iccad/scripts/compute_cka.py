#!/usr/bin/env python3
"""Compute per-layer CKA importance from llama-imatrix-hsic output.

Reads the binary probe files (input, per-layer, last_layer) and computes:
  S_in(i)  = CKA(X, H_i)   — similarity between input and layer i
  S_out(i) = CKA(H_i, H_L) — similarity between layer i and last layer

Metrics:
  M1 = S_out × (1 - S_in)   — product form (requires both conditions)
  M2 = S_out / S_in          — ratio form
  M3 = S_out - S_in          — difference form

Usage:
    python3 compute_cka.py qwen3/hsic/qwen3-hsic-q8_0
    python3 compute_cka.py qwen3/hsic/qwen3-hsic-q8_0 -o results.txt
    python3 compute_cka.py qwen3/hsic/qwen3-hsic-q8_0 --bootstrap 1000
"""

import argparse
import json
import sys

import numpy as np
from scipy.stats import spearmanr


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """CKA with linear kernel. X, Y: [n_samples, n_features].

    When n_samples < n_features, uses kernel-space formulation:
      K = X_c @ X_c.T, L = Y_c @ Y_c.T  (both [n, n])
      CKA = <K, L>_F / sqrt(<K, K>_F * <L, L>_F)
    which avoids computing [p, p] feature-space matrices.
    """
    n, p = X.shape
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    if n < p:
        # Kernel space: [n, n] matrices instead of [p, p]
        K = X @ X.T
        L = Y @ Y.T
        hsic_xy = np.sum(K * L)
        hsic_xx = np.sum(K * K)
        hsic_yy = np.sum(L * L)
    else:
        hsic_xy = np.linalg.norm(Y.T @ X, "fro") ** 2
        hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
        hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def rank_desc(arr):
    """Rank array in descending order (1 = highest value)."""
    order = np.argsort(-arr)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(arr))
    return ranks + 1


def main():
    parser = argparse.ArgumentParser(description="Compute CKA layer importance")
    parser.add_argument("prefix", help="Output prefix from llama-imatrix-hsic")
    parser.add_argument("-o", "--output", help="Write results to file (default: stdout only)")
    parser.add_argument("--bootstrap", type=int, default=0, metavar="N",
                        help="Run N bootstrap resamples to assess M1 ranking stability")
    args = parser.parse_args()

    meta_path = f"{args.prefix}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    n_c, n_e, n_l = meta["n_chunks"], meta["n_embd"], meta["n_layers"]

    X = np.fromfile(f"{args.prefix}_input.bin", dtype=np.float32).reshape(n_c, n_e)
    HL = np.fromfile(f"{args.prefix}_last_layer.bin", dtype=np.float32).reshape(n_c, n_e)

    s_in = np.zeros(n_l)
    s_out = np.zeros(n_l)
    for il in range(n_l):
        Hi = np.fromfile(f"{args.prefix}_layer_{il:02d}.bin", dtype=np.float32).reshape(n_c, n_e)
        s_in[il] = linear_CKA(X, Hi)
        s_out[il] = linear_CKA(Hi, HL)

    # Metrics
    m1 = s_out * (1.0 - s_in)
    m2 = s_out / s_in
    m3 = s_out - s_in

    # Exclude last layer from ranking (it is the reference for S_out, so S_out(L) = 1.0 trivially)
    last = n_l - 1
    rank_layers = [i for i in range(n_l) if i != last]

    r1, r2, r3 = rank_desc(m1[rank_layers]), rank_desc(m2[rank_layers]), rank_desc(m3[rank_layers])
    # Map back: rank_layers[idx] -> rank
    r1_full = np.full(n_l, -1, dtype=int)
    r2_full = np.full(n_l, -1, dtype=int)
    r3_full = np.full(n_l, -1, dtype=int)
    for idx, il in enumerate(rank_layers):
        r1_full[il] = r1[idx]
        r2_full[il] = r2[idx]
        r3_full[il] = r3[idx]

    # Build output
    lines = []
    lines.append(f"CKA Layer Importance Analysis")
    lines.append(f"Source: {args.prefix}")
    lines.append(f"Model:  {n_l} layers, n_embd={n_e}, {n_c} chunks, ctx={meta.get('n_ctx', 'N/A')}")
    lines.append(f"Note:   Layer {last} excluded from ranking (reference layer for S_out)")
    lines.append("")

    lines.append("=" * 90)
    lines.append("Per-Layer Results")
    lines.append("=" * 90)
    hdr = f"{'L':>3} | {'S_in':>7} | {'S_out':>7} | {'M1':>7} {'R1':>3} | {'M2':>7} {'R2':>3} | {'M3':>7} {'R3':>3}"
    lines.append(hdr)
    sub = f"{'':>3} | {'':>7} | {'':>7} | {'Sout(1-Sin)':>11} | {'Sout/Sin':>11} | {'Sout-Sin':>11}"
    lines.append(sub)
    lines.append("-" * 90)
    for il in range(n_l):
        if il == last:
            lines.append(
                f"{il:>3} | {s_in[il]:>7.4f} | {s_out[il]:>7.4f} "
                f"| {m1[il]:>7.4f}   * | {m2[il]:>7.4f}   * | {m3[il]:>7.4f}   *"
            )
        else:
            lines.append(
                f"{il:>3} | {s_in[il]:>7.4f} | {s_out[il]:>7.4f} "
                f"| {m1[il]:>7.4f} {r1_full[il]:>3} | {m2[il]:>7.4f} {r2_full[il]:>3} | {m3[il]:>7.4f} {r3_full[il]:>3}"
            )

    lines.append(f"  * = reference layer (excluded from ranking)")

    lines.append("")
    lines.append("=" * 90)
    lines.append("Summary Statistics (layers 0-{})".format(last - 1))
    lines.append("=" * 90)
    s_in_r, s_out_r = s_in[rank_layers], s_out[rank_layers]
    m1_r, m2_r, m3_r = m1[rank_layers], m2[rank_layers], m3[rank_layers]
    lines.append(f"S_in  range: [{s_in_r.min():.4f}, {s_in_r.max():.4f}]  mean={s_in_r.mean():.4f}  std={s_in_r.std():.4f}")
    lines.append(f"S_out range: [{s_out_r.min():.4f}, {s_out_r.max():.4f}]  mean={s_out_r.mean():.4f}  std={s_out_r.std():.4f}")
    lines.append(f"M1    range: [{m1_r.min():.4f}, {m1_r.max():.4f}]")
    lines.append(f"M2    range: [{m2_r.min():.4f}, {m2_r.max():.4f}]")
    lines.append(f"M3    range: [{m3_r.min():.4f}, {m3_r.max():.4f}]")

    n_show = 12
    lines.append("")
    lines.append("=" * 90)
    lines.append(f"Rankings — top/bottom {n_show} (1 = most important, layer {last} excluded)")
    lines.append("=" * 90)
    for name, m in [("M1 Sout*(1-Sin)", m1_r), ("M2 Sout/Sin", m2_r), ("M3 Sout-Sin", m3_r)]:
        order_most = np.argsort(-m)[:n_show]
        order_least = np.argsort(m)[:n_show]
        most = [rank_layers[int(x)] for x in order_most]
        least = [rank_layers[int(x)] for x in order_least]
        lines.append(f"{name:>16} — top {n_show}:    {most}")
        lines.append(f"{'':>16}   bottom {n_show}: {least}")

    lines.append("")
    lines.append("=" * 90)
    lines.append(f"Metric Correlations (Spearman rho, layers 0-{last - 1})")
    lines.append("=" * 90)
    metrics = {"M1": m1_r, "M2": m2_r, "M3": m3_r}
    names = list(metrics.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho, _ = spearmanr(metrics[names[i]], metrics[names[j]])
            lines.append(f"  {names[i]} vs {names[j]}: rho = {rho:.3f}")

    # Bootstrap analysis
    if args.bootstrap > 0:
        # Load all layer data into memory for resampling
        all_H = []
        for il in range(n_l):
            all_H.append(np.fromfile(f"{args.prefix}_layer_{il:02d}.bin", dtype=np.float32).reshape(n_c, n_e))

        rng = np.random.default_rng(42)
        n_boot = args.bootstrap
        n_sub = int(n_c * 0.8)
        boot_m1 = np.zeros((n_boot, n_l))

        for b in range(n_boot):
            idx = rng.choice(n_c, size=n_sub, replace=True)
            Xb, HLb = X[idx], HL[idx]
            for il in range(n_l):
                Hb = all_H[il][idx]
                si = linear_CKA(Xb, Hb)
                so = linear_CKA(Hb, HLb)
                boot_m1[b, il] = so * (1.0 - si)

        # Rank each bootstrap sample (exclude last layer)
        n_rank = len(rank_layers)
        boot_ranks = np.zeros((n_boot, n_rank), dtype=int)
        for b in range(n_boot):
            order = np.argsort(-boot_m1[b, rank_layers])
            for r, idx in enumerate(order):
                boot_ranks[b, idx] = r + 1

        lines.append("")
        lines.append("=" * 90)
        lines.append(f"Bootstrap Analysis ({n_boot} resamples, 80% chunks each)")
        lines.append("=" * 90)
        lines.append(f"{'L':>3} | {'M1 mean':>8} {'± std':>8} | {'med rank':>8} {'rank std':>8} | top-{n_show} freq")
        lines.append("-" * 90)

        # How often each layer appears in top-N
        n_top = n_show  # same as display count (12 for Qwen3, 10 for Qwen3.5)
        for ri, il in enumerate(rank_layers):
            m1_mean = boot_m1[:, il].mean()
            m1_std = boot_m1[:, il].std()
            rank_med = np.median(boot_ranks[:, ri])
            rank_std = boot_ranks[:, ri].std()
            top_freq = (boot_ranks[:, ri] <= n_top).sum() / n_boot * 100
            lines.append(f"{il:>3} | {m1_mean:>8.4f} {m1_std:>8.4f} | {rank_med:>8.1f} {rank_std:>8.2f} | {top_freq:>6.1f}%")

        # Top-N stability summary
        orig_topN = [rank_layers[int(x)] for x in np.argsort(-m1[rank_layers])[:n_top]]
        lines.append("")
        lines.append(f"Original top-{n_top} layers: {orig_topN}")
        # Count how often the exact original set appears
        orig_set = set(orig_topN)
        exact_match = 0
        overlap_counts = []
        for b in range(n_boot):
            boot_topN = set(rank_layers[int(x)] for x in np.argsort(-boot_m1[b, rank_layers])[:n_top])
            overlap_counts.append(len(orig_set & boot_topN))
            if boot_topN == orig_set:
                exact_match += 1
        lines.append(f"Exact match freq: {exact_match/n_boot*100:.1f}%")
        lines.append(f"Mean overlap with original: {np.mean(overlap_counts):.1f}/{n_top}")

    text = "\n".join(lines) + "\n"

    print(text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"(saved to {args.output})")


if __name__ == "__main__":
    main()
