#!/usr/bin/env python3
"""Pilot cross-linguistic analysis of triphone distinctive-feature distances."""

import os
import re
from collections import Counter, defaultdict
import numpy as np
from scipy import stats
import panphon
import epitran
import cmudict

# Standard ARPAbet (stress-stripped) to IPA mapping for CMUdict
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ",
    "AY": "aɪ", "B": "b", "CH": "tʃ", "D": "d", "DH": "ð",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "F": "f", "G": "ɡ",
    "HH": "h", "IH": "ɪ", "IY": "i", "JH": "dʒ", "K": "k",
    "L": "l", "M": "m", "N": "n", "NG": "ŋ", "OW": "oʊ",
    "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v",
    "W": "w", "Y": "j", "Z": "z", "ZH": "ʒ",
}

LANGUAGES = {
    "en": ("English", "cmudict"),
    "pt": ("Portuguese", "por-Latn"),
    "es": ("Spanish", "spa-Latn"),
    "de": ("German", "deu-Latn"),
    "tr": ("Turkish", "tur-Latn"),
    "ru": ("Russian", "rus-Cyrl"),
}


def load_ud_word_counts(conllu_path: str) -> Counter:
    """Extract lowercase word token counts from a UD .conllu file."""
    counts = Counter()
    with open(conllu_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#") or line.strip() == "":
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            tok_id, form, _, upos = cols[0], cols[1], cols[2], cols[3]
            if "-" in tok_id or "." in tok_id or upos == "PUNCT":
                continue
            word = form.lower()
            if re.search(r"[^\W\d_]", word):
                counts[word] += 1
    return counts


class FeatureDistanceCache:
    """Caches PanPhon feature vectors and pairwise Hamming distances."""

    def __init__(self):
        self.ft = panphon.FeatureTable()
        self.vec_cache = {}
        self.dist_cache = {}

    def seg_to_vec(self, seg: str):
        if seg not in self.vec_cache:
            vecs = self.ft.word_to_vector_list(seg, numeric=True)
            self.vec_cache[seg] = np.array(vecs[0], dtype=np.int8) if vecs else None
        return self.vec_cache[seg]

    def dist(self, seg1: str, seg2: str):
        key = (seg1, seg2)
        if key not in self.dist_cache:
            v1 = self.seg_to_vec(seg1)
            v2 = self.seg_to_vec(seg2)
            if v1 is None or v2 is None:
                d = None
            else:
                # Number of non-shared distinctive features (Hamming distance)
                d = int(np.sum(v1 != v2))
            self.dist_cache[key] = d
            self.dist_cache[(seg2, seg1)] = d
        return self.dist_cache[key]

    def ipa_to_segs(self, ipa: str):
        segs = self.ft.ipa_segs(ipa)
        return [s for s in segs if self.seg_to_vec(s) is not None]


def build_transcriber(mode: str, fdc: FeatureDistanceCache):
    if mode == "cmudict":
        cmu = cmudict.dict()

        def transcribe(word: str):
            phones = cmu.get(word)
            if not phones:
                return []
            segs = []
            for p in phones[0]:
                base = re.sub(r"\d+$", "", p)
                ipa = ARPABET_TO_IPA.get(base)
                if ipa:
                    segs.extend(fdc.ipa_to_segs(ipa))
            return segs

        return transcribe
    else:
        epi = epitran.Epitran(mode)

        def transcribe(word: str):
            ipa = epi.transliterate(word)
            return fdc.ipa_to_segs(ipa)

        return transcribe


def analyze_language(lang_code: str, lang_name: str, mode: str, conllu_path: str, fdc: FeatureDistanceCache):
    word_counts = load_ud_word_counts(conllu_path)
    transcribe = build_transcriber(mode, fdc)

    triphone_dist_counts = Counter()
    middle_d1 = defaultdict(float)
    middle_d2 = defaultdict(float)
    middle_w = defaultdict(float)

    word_means = []
    word_stds = []

    total_tokens = 0
    transcribed_tokens = 0
    total_triphones = 0

    for word, count in word_counts.items():
        total_tokens += count
        segs = transcribe(word)
        if not segs:
            continue
        transcribed_tokens += count

        if len(segs) >= 3:
            dists = [fdc.dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
            if all(d is not None for d in dists):
                word_means.append(np.mean(dists))
                word_stds.append(np.std(dists))
                for i in range(len(segs) - 2):
                    d1, d2 = dists[i], dists[i + 1]
                    mid = segs[i + 1]
                    triphone_dist_counts[(d1, d2)] += count
                    middle_d1[mid] += count * d1
                    middle_d2[mid] += count * d2
                    middle_w[mid] += count
                    total_triphones += count

    # Peak (d1, d2)
    top_pair, top_count = triphone_dist_counts.most_common(1)[0]
    d1_vals = [d1 for (d1, _), c in triphone_dist_counts.items() for _ in [0]]
    mean_d1 = sum(d1 * c for (d1, _), c in triphone_dist_counts.items()) / total_triphones
    mean_d2 = sum(d2 * c for (_, d2), c in triphone_dist_counts.items()) / total_triphones

    # Middle-phone asymmetry
    mids = [m for m, w in middle_w.items() if w >= 50]
    d1_mid = np.array([middle_d1[m] / middle_w[m] for m in mids])
    d2_mid = np.array([middle_d2[m] / middle_w[m] for m in mids])
    frac_d1_gt_d2 = np.mean(d1_mid > d2_mid) if len(mids) else np.nan
    wilcoxon_p = stats.wilcoxon(d1_mid, d2_mid).pvalue if len(mids) >= 5 else np.nan

    # Word-level mean vs std correlation
    r_word, p_word = stats.pearsonr(word_means, word_stds)

    return {
        "lang": lang_code,
        "name": lang_name,
        "types": len(word_counts),
        "tokens": total_tokens,
        "coverage": transcribed_tokens / total_tokens if total_tokens else 0,
        "triphones": total_triphones,
        "peak_d1_d2": top_pair,
        "peak_pct": 100.0 * top_count / total_triphones,
        "mean_d1": mean_d1,
        "mean_d2": mean_d2,
        "n_mids": len(mids),
        "frac_d1_gt_d2": frac_d1_gt_d2,
        "wilcoxon_p": wilcoxon_p,
        "word_r": r_word,
        "word_p": p_word,
    }


def main():
    fdc = FeatureDistanceCache()
    results = []
    for code, (name, mode) in LANGUAGES.items():
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"Skipping {name} ({code}): file missing or empty")
            continue
        res = analyze_language(code, name, mode, path, fdc)
        results.append(res)
        print(
            f"{res['name']:10s} | cov={res['coverage']*100:5.1f}% | "
            f"peak={str(res['peak_d1_d2']):7s} ({res['peak_pct']:4.1f}%) | "
            f"E[d1]={res['mean_d1']:.2f}, E[d2]={res['mean_d2']:.2f} | "
            f"mid(d1>d2)={res['frac_d1_gt_d2']*100:5.1f}% (n={res['n_mids']}, p={res['wilcoxon_p']:.2e}) | "
            f"word r={res['word_r']:.3f} (p={res['word_p']:.2e})"
        )


if __name__ == "__main__":
    main()
