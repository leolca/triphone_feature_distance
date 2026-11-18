#!/usr/bin/env python3
"""Quantify major-class vs. finer feature contributions to adjacent distance,
and audit IPA segmentation (multi-char merged units and skipped characters)."""

import os
import numpy as np
import pandas as pd
import epitran
import panphon

from analyze_crosslinguistic import LANGUAGES, FeatureDistanceCache, load_ud_word_counts


def main():
    fdc = FeatureDistanceCache()
    ft = fdc.ft
    rows = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path):
            continue
        word_counts = load_ud_word_counts(path)
        epi = epitran.Epitran(mode)

        tot_raw_chars = 0
        tot_kept_chars = 0
        tot_segs = 0
        multi_char_tokens = 0
        unique_segs = set()
        multi_char_types = set()

        sum_dmaj = 0.0
        sum_dfin = 0.0
        tot_adj = 0

        for word, count in word_counts.items():
            ipa = epi.transliterate(word)
            segs = fdc.ipa_to_segs(ipa)
            tot_raw_chars += count * len(ipa)
            tot_kept_chars += count * sum(len(s) for s in segs)
            tot_segs += count * len(segs)
            for s in segs:
                unique_segs.add(s)
                if len(s) > 1:
                    multi_char_types.add(s)
                    multi_char_tokens += count
            if len(segs) >= 3:
                vecs = [fdc.seg_to_vec(s) for s in segs]
                for i in range(len(vecs) - 1):
                    v1, v2 = vecs[i], vecs[i + 1]
                    sum_dmaj += count * int(np.sum(v1[:3] != v2[:3]))
                    sum_dfin += count * int(np.sum(v1[3:] != v2[3:]))
                    tot_adj += count

        ed_maj = sum_dmaj / tot_adj
        ed_fin = sum_dfin / tot_adj
        ed_tot = ed_maj + ed_fin
        rows.append({
            "code": code,
            "language": name,
            "unique_segs": len(unique_segs),
            "multi_char_types": len(multi_char_types),
            "multi_char_tok_pct": 100.0 * multi_char_tokens / tot_segs,
            "skipped_char_pct": 100.0 * max(0, tot_raw_chars - tot_kept_chars) / max(1, tot_raw_chars),
            "Ed_major": ed_maj,
            "Ed_finer": ed_fin,
            "Ed_total": ed_tot,
            "major_share_pct": 100.0 * ed_maj / ed_tot,
            "finer_share_pct": 100.0 * ed_fin / ed_tot,
        })
        print(
            f"{name:10s} | multi={len(multi_char_types):2d}/{len(unique_segs):2d} ({100*multi_char_tokens/tot_segs:5.2f}%) | "
            f"skip={100*max(0,tot_raw_chars-tot_kept_chars)/max(1,tot_raw_chars):4.2f}% | "
            f"E[d_maj]={ed_maj:.2f} ({100*ed_maj/ed_tot:4.1f}%), E[d_fin]={ed_fin:.2f} ({100*ed_fin/ed_tot:4.1f}%)"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/major_vs_finer_and_audit.csv", index=False)


if __name__ == "__main__":
    main()
