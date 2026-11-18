#!/usr/bin/env python3
"""Sensitivity checks: cosine binning (10/20/40), PanPhon weighted & partial Hamming,
non-major-class feature ablation (beyond C/V), IPA segmentation audit, and 5-family aggregation."""

import os
import numpy as np
import pandas as pd
from scipy import stats
import panphon.distance

from analyze_crosslinguistic import (
    LANGUAGES,
    FeatureDistanceCache,
    load_ud_word_counts,
    build_transcriber,
)
from analyze_cosine import CosineDistanceCache, compute_cosine_metrics
from analyze_cv_baseline import cv_corpus_shuffle


def eval_custom_dist(word_seg_counts, dist_fn, n_bins=20, max_val=None):
    """Compute E[d], std(d), and binned/discrete joint entropy H(d1, d2) for any distance fn."""
    sum_d = 0.0
    sum_d2 = 0.0
    tot_adj = 0
    if max_val is None:
        # Use Counter over exact values (rounded to 6 decimals)
        from collections import Counter
        mat_counts = Counter()
        for segs, count in word_seg_counts:
            dists = [dist_fn(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
            for d in dists:
                sum_d += count * d
                sum_d2 += count * d * d
                tot_adj += count
            for i in range(len(segs) - 2):
                mat_counts[(round(dists[i], 6), round(dists[i + 1], 6))] += count
        tot_tri = sum(mat_counts.values())
        probs = np.array(list(mat_counts.values()), dtype=np.float64) / tot_tri
    else:
        mat = np.zeros((n_bins, n_bins), dtype=np.float64)
        for segs, count in word_seg_counts:
            dists = [dist_fn(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
            for d in dists:
                sum_d += count * d
                sum_d2 += count * d * d
                tot_adj += count
            for i in range(len(segs) - 2):
                b1 = min(int((dists[i] / max_val) * n_bins), n_bins - 1)
                b2 = min(int((dists[i + 1] / max_val) * n_bins), n_bins - 1)
                mat[b1, b2] += count
        probs = mat[mat > 0] / mat.sum()

    mean_d = sum_d / tot_adj
    std_d = float(np.sqrt(max(0.0, sum_d2 / tot_adj - mean_d * mean_d)))
    H = float(-np.sum(probs * np.log2(probs)))
    return mean_d, std_d, H


def main():
    fdc = FeatureDistanceCache()
    cdc = CosineDistanceCache()
    ppd = panphon.distance.Distance()

    # Caches for weighted, partial Hamming, and non-major-class (indices 3..18)
    w_cache, ph_cache, sub_cache = {}, {}, {}

    def d_weighted(s1, s2):
        k = (s1, s2)
        if k not in w_cache:
            w_cache[k] = w_cache[(s2, s1)] = float(ppd.weighted_feature_edit_distance(s1, s2))
        return w_cache[k]

    def d_partial(s1, s2):
        k = (s1, s2)
        if k not in ph_cache:
            ph_cache[k] = ph_cache[(s2, s1)] = float(ppd.partial_hamming_feature_edit_distance(s1, s2))
        return ph_cache[k]

    def d_nonmajor(s1, s2):
        k = (s1, s2)
        if k not in sub_cache:
            v1 = fdc.seg_to_vec(s1)[3:19]
            v2 = fdc.seg_to_vec(s2)[3:19]
            sub_cache[k] = sub_cache[(s2, s1)] = int(np.sum(v1 != v2))
        return sub_cache[k]

    rows = []
    seg_audit_rows = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path):
            continue
        word_counts = load_ud_word_counts(path)
        transcribe = build_transcriber(mode, fdc)

        word_seg_counts = []
        unique_segs = set()
        multi_char_segs = 0
        total_seg_tokens = 0
        multi_char_tokens = 0
        for word, count in word_counts.items():
            segs = transcribe(word)
            if segs:
                unique_segs.update(segs)
                total_seg_tokens += count * len(segs)
                multi_char_tokens += count * sum(1 for s in segs if len(s) > 1)
                if len(segs) >= 3:
                    word_seg_counts.append((segs, count))

        multi_char_segs = sum(1 for s in unique_segs if len(s) > 1)
        seg_audit_rows.append({
            "code": code,
            "language": name,
            "unique_segs": len(unique_segs),
            "multi_char_types": multi_char_segs,
            "multi_char_tok_pct": 100.0 * multi_char_tokens / total_seg_tokens,
        })

        # Unrest and CV baselines
        rng_u = np.random.default_rng(42)
        unrest_words = []
        for segs, c in word_seg_counts:
            s2 = list(segs)
            rng_u.shuffle(s2)
            unrest_words.append((s2, c))
        cv_words = cv_corpus_shuffle(word_seg_counts, fdc, np.random.default_rng(42))

        # 1. Cosine binning sensitivity (10, 20, 40)
        c10_r = compute_cosine_metrics(word_seg_counts, cdc, n_bins=10, shuffle=False)["entropy_d1_d2"]
        c10_s = compute_cosine_metrics(word_seg_counts, cdc, n_bins=10, shuffle=True, rng=np.random.default_rng(42))["entropy_d1_d2"]
        c40_r = compute_cosine_metrics(word_seg_counts, cdc, n_bins=40, shuffle=False)["entropy_d1_d2"]
        c40_s = compute_cosine_metrics(word_seg_counts, cdc, n_bins=40, shuffle=True, rng=np.random.default_rng(42))["entropy_d1_d2"]

        # 2. Weighted, Partial Hamming, Non-major-class
        w_ed_r, w_sd_r, w_H_r = eval_custom_dist(word_seg_counts, d_weighted, n_bins=20, max_val=12.0)
        w_ed_c, w_sd_c, w_H_c = eval_custom_dist(cv_words, d_weighted, n_bins=20, max_val=12.0)
        w_ed_u, w_sd_u, w_H_u = eval_custom_dist(unrest_words, d_weighted, n_bins=20, max_val=12.0)

        p_ed_r, p_sd_r, p_H_r = eval_custom_dist(word_seg_counts, d_partial)
        p_ed_c, p_sd_c, p_H_c = eval_custom_dist(cv_words, d_partial)
        p_ed_u, p_sd_u, p_H_u = eval_custom_dist(unrest_words, d_partial)

        nm_ed_r, nm_sd_r, nm_H_r = eval_custom_dist(word_seg_counts, d_nonmajor)
        nm_ed_c, nm_sd_c, nm_H_c = eval_custom_dist(cv_words, d_nonmajor)
        nm_ed_u, nm_sd_u, nm_H_u = eval_custom_dist(unrest_words, d_nonmajor)

        rows.append({
            "code": code, "language": name,
            "family_top": family.split(" (")[0],
            "c10_r": c10_r, "c10_s": c10_s, "c40_r": c40_r, "c40_s": c40_s,
            "w_ed_r": w_ed_r, "w_ed_c": w_ed_c, "w_ed_u": w_ed_u,
            "w_sd_r": w_sd_r, "w_sd_c": w_sd_c, "w_sd_u": w_sd_u,
            "w_H_r": w_H_r, "w_H_c": w_H_c, "w_H_u": w_H_u,
            "p_ed_r": p_ed_r, "p_ed_c": p_ed_c, "p_ed_u": p_ed_u,
            "p_sd_r": p_sd_r, "p_sd_c": p_sd_c, "p_sd_u": p_sd_u,
            "p_H_r": p_H_r, "p_H_c": p_H_c, "p_H_u": p_H_u,
            "nm_ed_r": nm_ed_r, "nm_ed_c": nm_ed_c, "nm_ed_u": nm_ed_u,
            "nm_sd_r": nm_sd_r, "nm_sd_c": nm_sd_c, "nm_sd_u": nm_sd_u,
            "nm_H_r": nm_H_r, "nm_H_c": nm_H_c, "nm_H_u": nm_H_u,
        })
        print(
            f"{name:10s} | w_H r={w_H_r:.2f} c={w_H_c:.2f} u={w_H_u:.2f} | "
            f"p_H r={p_H_r:.2f} c={p_H_c:.2f} u={p_H_u:.2f} | "
            f"nonmajor_H r={nm_H_r:.2f} c={nm_H_c:.2f} u={nm_H_u:.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/feature_sensitivity.csv", index=False)
    pd.DataFrame(seg_audit_rows).to_csv("results/segmentation_audit.csv", index=False)

    print("\n--- Summary Checks across n=13 ---")
    print("Cosine 10-bin H_real < H_shuf:", int((df["c10_r"] < df["c10_s"]).sum()), "/ 13")
    print("Cosine 40-bin H_real < H_shuf:", int((df["c40_r"] < df["c40_s"]).sum()), "/ 13")

    for prefix, title in [("w", "PanPhon Weighted"), ("p", "Partial Hamming"), ("nm", "Non-Major (16 feat)")]:
        for vs, sfx in [("Unrest", "u"), ("CV", "c")]:
            ed_gt = int((df[f"{prefix}_ed_r"] > df[f"{prefix}_ed_{sfx}"]).sum())
            sd_lt = int((df[f"{prefix}_sd_r"] < df[f"{prefix}_sd_{sfx}"]).sum())
            H_lt = int((df[f"{prefix}_H_r"] < df[f"{prefix}_H_{sfx}"]).sum())
            _, p_ed = stats.wilcoxon(df[f"{prefix}_ed_r"], df[f"{prefix}_ed_{sfx}"])
            _, p_sd = stats.wilcoxon(df[f"{prefix}_sd_r"], df[f"{prefix}_sd_{sfx}"])
            _, p_H = stats.wilcoxon(df[f"{prefix}_H_r"], df[f"{prefix}_H_{sfx}"])
            print(f"{title:20s} vs {vs:6s}: E[d]> ({ed_gt}/13, p={p_ed:.2e}) | sd< ({sd_lt}/13, p={p_sd:.2e}) | H< ({H_lt}/13, p={p_H:.2e})")

    # 5-family equal-weight aggregation on main summary & cv_baseline
    dh = pd.read_csv("results/summary.csv")
    cv = pd.read_csv("results/cv_baseline.csv")
    dh["family_top"] = dh["family"].str.split(r" \(").str[0]
    cv["family_top"] = dh["family_top"]
    fam_dh = dh.groupby("family_top")[["mean_d_real", "mean_d_shuf", "std_d_real", "std_d_shuf", "H_d1d2_real", "H_d1d2_shuf"]].mean()
    fam_cv = cv.groupby("family_top")[["Ed_real", "Ed_cv", "sd_real", "sd_cv", "H_real", "H_cv"]].mean()
    print("\n--- 5-Family Equal-Weight Means (Real vs Unrest) ---")
    print(fam_dh.round(3))
    print("\n--- 5-Family Equal-Weight Means (Real vs CV) ---")
    print(fam_cv.round(3))


if __name__ == "__main__":
    main()
