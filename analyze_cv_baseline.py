#!/usr/bin/env python3
"""Test CV-skeleton-preserving baseline and masked-Hamming (ignoring 0s) across 13 languages."""

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


def cv_corpus_shuffle(word_seg_counts, fdc: FeatureDistanceCache, rng):
    """Preserve each word's exact C/V skeleton (syl == 1 vs syl != 1),
    permuting C segments across C slots and V segments across V slots."""
    c_pool = []
    v_pool = []
    skeletons = []

    for segs, count in word_seg_counts:
        skel = []
        for s in segs:
            is_v = int(fdc.seg_to_vec(s)[0]) == 1
            skel.append(is_v)
            if is_v:
                v_pool.extend([s] * count)
            else:
                c_pool.extend([s] * count)
        skeletons.append((skel, count))

    rng.shuffle(c_pool)
    rng.shuffle(v_pool)

    # Reconstruct token-level words (or sample per word occurrence)
    # Build new word_seg_counts by consuming c_pool and v_pool
    c_idx = 0
    v_idx = 0
    new_counts = {}
    for skel, count in skeletons:
        for _ in range(count):
            w = []
            for is_v in skel:
                if is_v:
                    w.append(v_pool[v_idx])
                    v_idx += 1
                else:
                    w.append(c_pool[c_idx])
                    c_idx += 1
            wt = tuple(w)
            new_counts[wt] = new_counts.get(wt, 0) + 1
    return list(new_counts.items())


def main():
    fdc = FeatureDistanceCache()
    rng = np.random.default_rng(42)
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

        real = compute_metrics(word_seg_counts, fdc, shuffle=False)
        shuf_free = compute_metrics(word_seg_counts, fdc, shuffle=True, rng=np.random.default_rng(42))
        cv_words = cv_corpus_shuffle(word_seg_counts, fdc, rng)
        shuf_cv = compute_metrics(cv_words, fdc, shuffle=False)

        rows.append({
            "code": code,
            "language": name,
            "Ed_real": real["mean_d"],
            "Ed_cv": shuf_cv["mean_d"],
            "Ed_free": shuf_free["mean_d"],
            "sd_real": real["std_d"],
            "sd_cv": shuf_cv["std_d"],
            "sd_free": shuf_free["std_d"],
            "H_real": real["entropy_d1_d2"],
            "H_cv": shuf_cv["entropy_d1_d2"],
            "H_free": shuf_free["entropy_d1_d2"],
        })
        print(
            f"{name:10s} | E[d] real={real['mean_d']:.2f} cv={shuf_cv['mean_d']:.2f} free={shuf_free['mean_d']:.2f} | "
            f"sd real={real['std_d']:.2f} cv={shuf_cv['std_d']:.2f} free={shuf_free['std_d']:.2f} | "
            f"H real={real['entropy_d1_d2']:.2f} cv={shuf_cv['entropy_d1_d2']:.2f} free={shuf_free['entropy_d1_d2']:.2f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/cv_baseline.csv", index=False)

    print("\n--- Real vs CV-skeleton-preserving baseline (Wilcoxon across n=17) ---")
    for label, cr, cc in [("E[d]", "Ed_real", "Ed_cv"), ("std(d)", "sd_real", "sd_cv"), ("H(d1,d2)", "H_real", "H_cv")]:
        w, p = stats.wilcoxon(df[cr], df[cc])
        n_gt = int((df[cr] > df[cc]).sum())
        print(f"{label:10s}: real>cv in {n_gt}/17 | W={w}, p={p:.4e}")


if __name__ == "__main__":
    main()
