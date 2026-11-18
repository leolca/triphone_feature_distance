#!/usr/bin/env python3
"""Cross-linguistic analysis of triphone distinctive-feature cosine distance vs. shuffled baseline."""

import os
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_crosslinguistic import (
    LANGUAGES,
    FeatureDistanceCache,
    load_ud_word_counts,
    build_transcriber,
)


class CosineDistanceCache(FeatureDistanceCache):
    def dist(self, seg1: str, seg2: str):
        key = (seg1, seg2)
        if key not in self.dist_cache:
            v1 = self.seg_to_vec(seg1)
            v2 = self.seg_to_vec(seg2)
            if v1 is None or v2 is None:
                d = None
            else:
                v1f = v1.astype(np.float64)
                v2f = v2.astype(np.float64)
                n1 = np.linalg.norm(v1f)
                n2 = np.linalg.norm(v2f)
                if n1 == 0.0 or n2 == 0.0:
                    d = None
                else:
                    cos_sim = np.dot(v1f, v2f) / (n1 * n2)
                    d = float(0.5 * (1.0 - cos_sim))
            self.dist_cache[key] = d
            self.dist_cache[(seg2, seg1)] = d
        return self.dist_cache[key]


def compute_cosine_metrics(word_seg_counts, cdc: CosineDistanceCache, n_bins: int = 20, shuffle: bool = False, rng=None):
    mat = np.zeros((n_bins, n_bins), dtype=np.float64)
    sum_d = 0.0
    sum_d2 = 0.0
    tot_adj = 0
    sum_d1 = 0.0
    sum_d2_tri = 0.0
    tot_tri = 0
    word_means = []
    word_stds = []

    for segs, count in word_seg_counts:
        if shuffle:
            segs = list(segs)
            rng.shuffle(segs)
        dists = [cdc.dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        if any(d is None for d in dists):
            continue
        for d in dists:
            sum_d += count * d
            sum_d2 += count * (d * d)
            tot_adj += count
        word_means.append(np.mean(dists))
        word_stds.append(np.std(dists))
        for i in range(len(segs) - 2):
            d1, d2 = dists[i], dists[i + 1]
            b1 = min(int(d1 * n_bins), n_bins - 1)
            b2 = min(int(d2 * n_bins), n_bins - 1)
            mat[b1, b2] += count
            sum_d1 += count * d1
            sum_d2_tri += count * d2
            tot_tri += count

    mean_d = sum_d / tot_adj
    var_d = max(0.0, (sum_d2 / tot_adj) - mean_d * mean_d)
    std_d = float(np.sqrt(var_d))

    mat_prob = mat / mat.sum() if mat.sum() > 0 else mat
    nz = mat_prob[mat_prob > 0]
    entropy_d1_d2 = float(-np.sum(nz * np.log2(nz)))

    peak_idx = np.unravel_index(np.argmax(mat_prob), mat_prob.shape)
    peak_low = (peak_idx[0] / n_bins, peak_idx[1] / n_bins)
    peak_pct = 100.0 * float(mat_prob[peak_idx])

    r_word, _ = stats.pearsonr(word_means, word_stds)

    return {
        "peak_b1": peak_idx[0],
        "peak_b2": peak_idx[1],
        "peak_c1": (peak_idx[0] + 0.5) / n_bins,
        "peak_c2": (peak_idx[1] + 0.5) / n_bins,
        "peak_pct": peak_pct,
        "mean_d": mean_d,
        "std_d": std_d,
        "entropy_d1_d2": entropy_d1_d2,
        "mean_d1": sum_d1 / tot_tri,
        "mean_d2": sum_d2_tri / tot_tri,
        "word_r": float(r_word),
        "mat_prob": mat_prob,
    }


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    cdc = CosineDistanceCache()
    rng = np.random.default_rng(42)

    rows = []
    heatmaps = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            continue
        word_counts = load_ud_word_counts(path)
        transcribe = build_transcriber(mode, cdc)

        word_seg_counts = []
        for word, count in word_counts.items():
            segs = transcribe(word)
            if segs and len(segs) >= 3:
                word_seg_counts.append((segs, count))

        real = compute_cosine_metrics(word_seg_counts, cdc, n_bins=20, shuffle=False)
        shuf = compute_cosine_metrics(word_seg_counts, cdc, n_bins=20, shuffle=True, rng=rng)

        rows.append({
            "code": code,
            "language": name,
            "family": family,
            "peak_c1": real["peak_c1"],
            "peak_c2": real["peak_c2"],
            "peak_pct": real["peak_pct"],
            "mean_d_real": real["mean_d"],
            "mean_d_shuf": shuf["mean_d"],
            "std_d_real": real["std_d"],
            "std_d_shuf": shuf["std_d"],
            "H_d1d2_real": real["entropy_d1_d2"],
            "H_d1d2_shuf": shuf["entropy_d1_d2"],
            "mean_d1": real["mean_d1"],
            "mean_d2": real["mean_d2"],
            "word_r_real": real["word_r"],
            "word_r_shuf": shuf["word_r"],
        })
        heatmaps.append((name, real["mat_prob"]))

        print(
            f"{name:10s} | peak=({real['peak_c1']:.2f},{real['peak_c2']:.2f}) | "
            f"E[d] real={real['mean_d']:.3f} vs shuf={shuf['mean_d']:.3f} | "
            f"std[d] real={real['std_d']:.3f} vs shuf={shuf['std_d']:.3f} | "
            f"H real={real['entropy_d1_d2']:.2f} vs shuf={shuf['entropy_d1_d2']:.2f} | "
            f"r real={real['word_r']:.3f} vs shuf={shuf['word_r']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/summary_cosine.csv", index=False)

    n_langs = len(heatmaps)
    ncols = 4
    nrows = int(np.ceil(n_langs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, (name, mat_real) in enumerate(heatmaps):
        ax = axes_flat[idx]
        im = ax.imshow(mat_real * 100, origin="lower", cmap="viridis", aspect="auto", extent=[0, 1, 0, 1])
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("$d_{\\cos,2}$")
        ax.set_ylabel("$d_{\\cos,1}$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for idx in range(n_langs, len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.colorbar(im, ax=axes_flat[:n_langs], label="Probability (%)", shrink=0.6)
    fig.savefig("figures/triphone_heatmaps_cosine.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.38
    panels = [
        (axes[0], "mean_d_real", "mean_d_shuf", "Mean Cosine Distance $\\mathbb{E}[d_{\\cos}]$", "Cosine distance"),
        (axes[1], "std_d_real", "std_d_shuf", "Std. Dev. $\\sigma(d_{\\cos})$", "Std. Dev."),
        (axes[2], "H_d1d2_real", "H_d1d2_shuf", "Joint Entropy $H(d_{\\cos,1}, d_{\\cos,2})$", "Entropy (bits)"),
    ]
    for ax, col_r, col_s, title, ylabel in panels:
        ax.bar(x - w / 2, df[col_r], width=w, label="Natural text", color="#2b5c8f")
        ax.bar(x + w / 2, df[col_s], width=w, label="Within-word shuffled", color="#d97a38")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(df["code"], rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig("figures/real_vs_shuffled_cosine.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
