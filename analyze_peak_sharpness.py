#!/usr/bin/env python3
"""Test structural correlates of triphone peak sharpness across 13 languages."""

import os
import numpy as np
import pandas as pd
from scipy import stats
from analyze_crosslinguistic import (
    LANGUAGES,
    FeatureDistanceCache,
    load_ud_word_counts,
    build_transcriber,
)


def main():
    fdc = FeatureDistanceCache()
    rows = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path):
            continue
        word_counts = load_ud_word_counts(path)
        transcribe = build_transcriber(mode, fdc)

        seg_vocab = set()
        mat = np.zeros((25, 25), dtype=np.float64)
        tot_tri = 0
        cvc_vcv_tri = 0
        sum_len_w = 0
        tot_tok = 0

        for word, count in word_counts.items():
            segs = transcribe(word)
            if not segs:
                continue
            seg_vocab.update(segs)
            sum_len_w += count * len(segs)
            tot_tok += count
            if len(segs) < 3:
                continue
            syls = [int(fdc.seg_to_vec(s)[0]) for s in segs]  # 'syl' is index 0
            dists = [fdc.dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
            if any(d is None for d in dists):
                continue
            for i in range(len(segs) - 2):
                d1, d2 = dists[i], dists[i + 1]
                mat[d1, d2] += count
                tot_tri += count
                s1, s2, s3 = syls[i], syls[i + 1], syls[i + 2]
                if (s1 == -1 and s2 == 1 and s3 == -1) or (s1 == 1 and s2 == -1 and s3 == 1):
                    cvc_vcv_tri += count

        mat_p = mat / tot_tri
        nz = mat_p[mat_p > 0]
        H = float(-np.sum(nz * np.log2(nz)))
        p_idx = np.unravel_index(np.argmax(mat_p), mat_p.shape)
        peak_pct = 100.0 * float(mat_p[p_idx])
        r0, c0 = p_idx
        top3x3 = 100.0 * float(mat_p[max(0, r0 - 1):r0 + 2, max(0, c0 - 1):c0 + 2].sum())

        rows.append({
            "code": code,
            "language": name,
            "inv_size": len(seg_vocab),
            "mean_len": sum_len_w / tot_tok,
            "cvc_vcv_pct": 100.0 * cvc_vcv_tri / tot_tri,
            "peak_pct": peak_pct,
            "top3x3_pct": top3x3,
            "H_d1d2": H,
        })

    df = pd.DataFrame(rows).sort_values("peak_pct", ascending=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n--- Correlations with Peak Sharpness ---")
    for metric in ["peak_pct", "top3x3_pct", "H_d1d2"]:
        for pred in ["inv_size", "cvc_vcv_pct", "mean_len"]:
            r, p = stats.pearsonr(df[pred], df[metric])
            rho, sp_p = stats.spearmanr(df[pred], df[metric])
            print(f"{metric:10s} vs {pred:12s}: Pearson r={r:+.3f} (p={p:.3f}) | Spearman rho={rho:+.3f} (p={sp_p:.3f})")


if __name__ == "__main__":
    main()
