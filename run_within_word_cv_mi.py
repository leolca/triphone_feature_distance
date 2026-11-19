#!/usr/bin/env python3
"""Within-word C/V-preserving shuffle (20 seeds) + Mutual Information decomposition:
H(d1, d2) = H(d1) + H(d2) - I(d1; d2) across all 17 languages."""

import os
from collections import Counter
import numpy as np
import pandas as pd
from scipy import stats

from analyze_crosslinguistic import (
    LANGUAGES,
    FeatureDistanceCache,
    load_ud_word_counts,
    build_transcriber,
)


def entropy_and_mi(word_seg_counts, fdc: FeatureDistanceCache):
    c1 = Counter()
    c2 = Counter()
    c12 = Counter()
    sum_d = 0.0
    sum_d2 = 0.0
    tot_adj = 0

    for segs, count in word_seg_counts:
        dists = [fdc.dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        if any(d is None for d in dists):
            continue
        for d in dists:
            sum_d += count * d
            sum_d2 += count * d * d
            tot_adj += count
        for i in range(len(segs) - 2):
            d1, d2 = dists[i], dists[i + 1]
            c1[d1] += count
            c2[d2] += count
            c12[(d1, d2)] += count

    def H_from_counter(cnt):
        tot = sum(cnt.values())
        p = np.array(list(cnt.values()), dtype=np.float64) / tot
        return float(-np.sum(p * np.log2(p)))

    H1 = H_from_counter(c1)
    H2 = H_from_counter(c2)
    H12 = H_from_counter(c12)
    MI = H1 + H2 - H12
    Ed = sum_d / tot_adj
    sd = float(np.sqrt(max(0.0, sum_d2 / tot_adj - Ed * Ed)))
    return Ed, sd, H1, H2, H12, MI


def shuffle_within_word_free(word_seg_counts, rng):
    out = []
    for segs, count in word_seg_counts:
        s = list(segs)
        rng.shuffle(s)
        out.append((s, count))
    return out


def shuffle_within_word_cv(word_seg_counts, fdc: FeatureDistanceCache, rng):
    out = []
    for segs, count in word_seg_counts:
        c_idx = [i for i, s in enumerate(segs) if int(fdc.seg_to_vec(s)[0]) != 1]
        v_idx = [i for i, s in enumerate(segs) if int(fdc.seg_to_vec(s)[0]) == 1]
        c_segs = [segs[i] for i in c_idx]
        v_segs = [segs[i] for i in v_idx]
        rng.shuffle(c_segs)
        rng.shuffle(v_segs)
        s = list(segs)
        for pos, val in zip(c_idx, c_segs):
            s[pos] = val
        for pos, val in zip(v_idx, v_segs):
            s[pos] = val
        out.append((s, count))
    return out


def main():
    fdc = FeatureDistanceCache()
    rows = []

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

        Ed_r, sd_r, H1_r, H2_r, H12_r, MI_r = entropy_and_mi(word_seg_counts, fdc)

        cv_H12, cv_MI, cv_Ed, cv_sd, cv_H1, cv_H2 = [], [], [], [], [], []
        fr_H12, fr_MI = [], []
        for seed in range(20):
            rng_cv = np.random.default_rng(2000 + seed)
            rng_fr = np.random.default_rng(3000 + seed)
            e, s, h1, h2, h12, mi = entropy_and_mi(shuffle_within_word_cv(word_seg_counts, fdc, rng_cv), fdc)
            cv_Ed.append(e); cv_sd.append(s); cv_H1.append(h1); cv_H2.append(h2); cv_H12.append(h12); cv_MI.append(mi)
            _, _, _, _, _, mi_f = entropy_and_mi(shuffle_within_word_free(word_seg_counts, rng_fr), fdc)
            fr_H12.append(_); fr_MI.append(mi_f)

        rows.append({
            "code": code, "language": name, "family": family.split(" (")[0],
            "Ed_real": Ed_r, "Ed_cv": np.mean(cv_Ed), "Ed_cv_sd": np.std(cv_Ed, ddof=1),
            "sd_real": sd_r, "sd_cv": np.mean(cv_sd), "sd_cv_sd": np.std(cv_sd, ddof=1),
            "H1_real": H1_r, "H2_real": H2_r, "H1_cv": np.mean(cv_H1), "H2_cv": np.mean(cv_H2),
            "H12_real": H12_r, "H12_cv": np.mean(cv_H12), "H12_cv_sd": np.std(cv_H12, ddof=1),
            "MI_real": MI_r, "MI_cv": np.mean(cv_MI), "MI_cv_sd": np.std(cv_MI, ddof=1),
            "MI_free": np.mean(fr_MI),
        })
        print(
            f"{name:10s} | H12 real={H12_r:.3f} cv={np.mean(cv_H12):.3f}±{np.std(cv_H12,ddof=1):.3f} | "
            f"H1+H2 real={H1_r+H2_r:.3f} cv={np.mean(cv_H1)+np.mean(cv_H2):.3f} | "
            f"MI real={MI_r:.3f} cv={np.mean(cv_MI):.3f}±{np.std(cv_MI,ddof=1):.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/within_word_cv_mi.csv", index=False)

    print("\n--- Paired Wilcoxon across n=17 languages (Real vs Within-Word C/V Shuffle) ---")
    for label, cr, cc, expect_lt in [
        ("E[d]", "Ed_real", "Ed_cv", False),
        ("std(d)", "sd_real", "sd_cv", True),
        ("H1+H2 (marginal)", None, None, True),
        ("H(d1,d2)", "H12_real", "H12_cv", True),
        ("MI I(d1;d2)", "MI_real", "MI_cv", False),
    ]:
        if cr is None:
            a = df["H1_real"] + df["H2_real"]
            b = df["H1_cv"] + df["H2_cv"]
        else:
            a, b = df[cr], df[cc]
        w, p = stats.wilcoxon(a, b)
        n_dir = int((a < b).sum()) if expect_lt else int((a > b).sum())
        sym = "<" if expect_lt else ">"
        print(f"{label:18s}: real {sym} cv in {n_dir}/17 | W={w:.1f}, p={p:.4e}")


if __name__ == "__main__":
    main()
