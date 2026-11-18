#!/usr/bin/env python3
"""Monte Carlo multi-seed baseline, 17k-token subsampling, and type-weighted robustness checks."""

import os
import numpy as np
import pandas as pd
from scipy import stats

from analyze_crosslinguistic import (
    LANGUAGES,
    FeatureDistanceCache,
    load_ud_word_counts,
    build_transcriber,
    compute_metrics,
)


def main():
    fdc = FeatureDistanceCache()
    rng_sub = np.random.default_rng(123)

    mc_rows = []
    sub_rows = []
    type_rows = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path):
            continue
        word_counts = load_ud_word_counts(path)
        transcribe = build_transcriber(mode, fdc)

        word_seg_counts = []
        for word, count in word_counts.items():
            segs = transcribe(word)
            if segs and len(segs) >= 3:
                word_seg_counts.append((segs, count))

        real = compute_metrics(word_seg_counts, fdc, shuffle=False)

        # 1. 20-seed Monte Carlo shuffled baseline
        shuf_ed, shuf_sd, shuf_H = [], [], []
        for seed in range(20):
            rng = np.random.default_rng(1000 + seed)
            s = compute_metrics(word_seg_counts, fdc, shuffle=True, rng=rng)
            shuf_ed.append(s["mean_d"])
            shuf_sd.append(s["std_d"])
            shuf_H.append(s["entropy_d1_d2"])

        mc_rows.append({
            "code": code,
            "language": name,
            "real_Ed": real["mean_d"],
            "shuf_Ed_mean": np.mean(shuf_ed),
            "shuf_Ed_sd": np.std(shuf_ed, ddof=1),
            "real_sd": real["std_d"],
            "shuf_sd_mean": np.mean(shuf_sd),
            "shuf_sd_sd": np.std(shuf_sd, ddof=1),
            "real_H": real["entropy_d1_d2"],
            "shuf_H_mean": np.mean(shuf_H),
            "shuf_H_sd": np.std(shuf_H, ddof=1),
        })

        # 2. Type-weighted (count = 1 per unique word type)
        type_seg_counts = [(segs, 1) for segs, _ in word_seg_counts]
        t_real = compute_metrics(type_seg_counts, fdc, shuffle=False)
        t_shuf = compute_metrics(type_seg_counts, fdc, shuffle=True, rng=np.random.default_rng(42))
        type_rows.append({
            "code": code,
            "language": name,
            "type_Ed_real": t_real["mean_d"],
            "type_Ed_shuf": t_shuf["mean_d"],
            "type_sd_real": t_real["std_d"],
            "type_sd_shuf": t_shuf["std_d"],
            "type_H_real": t_real["entropy_d1_d2"],
            "type_H_shuf": t_shuf["entropy_d1_d2"],
        })

        # 3. Subsampled to 17,000 tokens (matching smallest corpus, Hungarian)
        # Expand token list of (word_idx) and sample up to 17,000
        all_weights = np.array([c for _, c in word_seg_counts], dtype=np.int64)
        tot_w = int(all_weights.sum())
        if tot_w > 17000:
            probs = all_weights / tot_w
            sampled_counts = rng_sub.multinomial(17000, probs)
            sub_seg_counts = [(segs, int(sc)) for (segs, _), sc in zip(word_seg_counts, sampled_counts) if sc > 0]
        else:
            sub_seg_counts = word_seg_counts
        s_real = compute_metrics(sub_seg_counts, fdc, shuffle=False)
        s_shuf = compute_metrics(sub_seg_counts, fdc, shuffle=True, rng=np.random.default_rng(42))
        sub_rows.append({
            "code": code,
            "language": name,
            "sub_Ed_real": s_real["mean_d"],
            "sub_Ed_shuf": s_shuf["mean_d"],
            "sub_sd_real": s_real["std_d"],
            "sub_sd_shuf": s_shuf["std_d"],
            "sub_H_real": s_real["entropy_d1_d2"],
            "sub_H_shuf": s_shuf["entropy_d1_d2"],
        })
        print(f"Done {name}")

    df_mc = pd.DataFrame(mc_rows)
    df_type = pd.DataFrame(type_rows)
    df_sub = pd.DataFrame(sub_rows)

    df_mc.to_csv("results/robustness_mc20.csv", index=False)
    df_type.to_csv("results/robustness_type.csv", index=False)
    df_sub.to_csv("results/robustness_sub17k.csv", index=False)

    print("\n--- Cross-linguistic paired Wilcoxon & Sign tests (n=13) ---")
    for label, col_r, col_s in [
        ("MC20 E[d]", "real_Ed", "shuf_Ed_mean"),
        ("MC20 std(d)", "real_sd", "shuf_sd_mean"),
        ("MC20 H(d1,d2)", "real_H", "shuf_H_mean"),
    ]:
        w_stat, w_p = stats.wilcoxon(df_mc[col_r], df_mc[col_s])
        print(f"{label:15s}: Wilcoxon W={w_stat}, p={w_p:.6e}")

    print("\nType-weighted 13/13 direction check:")
    print("  E[d] real > shuf:", int((df_type["type_Ed_real"] > df_type["type_Ed_shuf"]).sum()), "/ 13")
    print("  std  real < shuf:", int((df_type["type_sd_real"] < df_type["type_sd_shuf"]).sum()), "/ 13")
    print("  H    real < shuf:", int((df_type["type_H_real"] < df_type["type_H_shuf"]).sum()), "/ 13")

    print("\nSubsampled 17k 13/13 direction check:")
    print("  E[d] real > shuf:", int((df_sub["sub_Ed_real"] > df_sub["sub_Ed_shuf"]).sum()), "/ 13")
    print("  std  real < shuf:", int((df_sub["sub_sd_real"] < df_sub["sub_sd_shuf"]).sum()), "/ 13")
    print("  H    real < shuf:", int((df_sub["sub_H_real"] < df_sub["sub_H_shuf"]).sum()), "/ 13")


if __name__ == "__main__":
    main()
