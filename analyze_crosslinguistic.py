#!/usr/bin/env python3
"""Cross-linguistic analysis of triphone distinctive-feature distances vs. shuffled baseline."""

import os
import re
from collections import Counter, defaultdict

# Ensure .venv/bin is on PATH so Epitran's eng-Latn finds lex_lookup
_venv_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "bin")
if os.path.isdir(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import panphon
import epitran
import cmudict

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

LANGUAGES = [
    ("en", "English", "Indo-European (Germanic)", "eng-Latn"),
    ("de", "German", "Indo-European (Germanic)", "deu-Latn"),
    ("pt", "Portuguese", "Indo-European (Romance)", "por-Latn"),
    ("es", "Spanish", "Indo-European (Romance)", "spa-Latn"),
    ("fr", "French", "Indo-European (Romance)", "fra-Latn"),
    ("pl", "Polish", "Indo-European (Slavic)", "pol-Latn"),
    ("ru", "Russian", "Indo-European (Slavic)", "rus-Cyrl"),
    ("hi", "Hindi", "Indo-European (Indo-Aryan)", "hin-Deva"),
    ("hu", "Hungarian", "Uralic", "hun-Latn"),
    ("fi", "Finnish", "Uralic", "fin-Latn"),
    ("tr", "Turkish", "Turkic", "tur-Latn"),
    ("id", "Indonesian", "Austronesian", "ind-Latn"),
    ("ko", "Korean", "Koreanic", "kor-Hang"),
    ("ta", "Tamil", "Dravidian", "tam-Taml"),
    ("mt", "Maltese", "Afro-Asiatic (Semitic)", "mlt-Latn"),
    ("vi", "Vietnamese", "Austroasiatic", "vie-Latn"),
    ("yo", "Yoruba", "Niger-Congo", "yor-Latn"),
]


def load_ud_word_counts(conllu_path: str) -> Counter:
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


def compute_metrics(word_seg_counts, fdc: FeatureDistanceCache, shuffle: bool = False, rng=None):
    triphone_dist_counts = Counter()
    adj_dist_counts = Counter()
    middle_d1 = defaultdict(float)
    middle_d2 = defaultdict(float)
    middle_w = defaultdict(float)
    word_means = []
    word_stds = []
    total_triphones = 0

    for segs, count in word_seg_counts:
        if shuffle:
            segs = list(segs)
            rng.shuffle(segs)
        dists = [fdc.dist(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        if any(d is None for d in dists):
            continue
        for d in dists:
            adj_dist_counts[d] += count
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

    # Joint distribution matrix (0..24 x 0..24)
    mat = np.zeros((25, 25), dtype=np.float64)
    for (d1, d2), c in triphone_dist_counts.items():
        if d1 < 25 and d2 < 25:
            mat[d1, d2] = c
    mat_prob = mat / mat.sum() if mat.sum() > 0 else mat

    # Shannon entropy (bits) of (d1, d2)
    nz = mat_prob[mat_prob > 0]
    entropy_d1_d2 = float(-np.sum(nz * np.log2(nz)))

    # Adjacent distance stats
    tot_adj = sum(adj_dist_counts.values())
    mean_d = sum(d * c for d, c in adj_dist_counts.items()) / tot_adj
    var_d = sum(((d - mean_d) ** 2) * c for d, c in adj_dist_counts.items()) / tot_adj
    std_d = float(np.sqrt(var_d))

    top_pair, top_count = triphone_dist_counts.most_common(1)[0]
    mean_d1 = sum(d1 * c for (d1, _), c in triphone_dist_counts.items()) / total_triphones
    mean_d2 = sum(d2 * c for (_, d2), c in triphone_dist_counts.items()) / total_triphones

    mids = [m for m, w in middle_w.items() if w >= 50]
    d1_mid = np.array([middle_d1[m] / middle_w[m] for m in mids])
    d2_mid = np.array([middle_d2[m] / middle_w[m] for m in mids])
    frac_d1_gt_d2 = float(np.mean(d1_mid > d2_mid)) if len(mids) else np.nan
    wilcoxon_p = float(stats.wilcoxon(d1_mid, d2_mid).pvalue) if len(mids) >= 5 else np.nan

    r_word, p_word = stats.pearsonr(word_means, word_stds)

    return {
        "triphones": total_triphones,
        "peak_d1": top_pair[0],
        "peak_d2": top_pair[1],
        "peak_pct": 100.0 * top_count / total_triphones,
        "mean_d": mean_d,
        "std_d": std_d,
        "entropy_d1_d2": entropy_d1_d2,
        "mean_d1": mean_d1,
        "mean_d2": mean_d2,
        "n_mids": len(mids),
        "frac_d1_gt_d2": frac_d1_gt_d2,
        "wilcoxon_p": wilcoxon_p,
        "word_r": float(r_word),
        "word_p": float(p_word),
        "mat_prob": mat_prob,
        "adj_counts": adj_dist_counts,
    }


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    fdc = FeatureDistanceCache()
    rng = np.random.default_rng(42)

    rows = []
    heatmaps = []

    for code, name, family, mode in LANGUAGES:
        path = os.path.join("data", "ud", f"{code}.conllu")
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"Skipping {name} ({code}): missing or empty")
            continue

        word_counts = load_ud_word_counts(path)
        transcribe = build_transcriber(mode, fdc)

        word_seg_counts = []
        total_tokens = 0
        transcribed_tokens = 0
        for word, count in word_counts.items():
            total_tokens += count
            segs = transcribe(word)
            if not segs:
                continue
            transcribed_tokens += count
            if len(segs) >= 3:
                word_seg_counts.append((segs, count))

        real = compute_metrics(word_seg_counts, fdc, shuffle=False)
        shuf = compute_metrics(word_seg_counts, fdc, shuffle=True, rng=rng)

        rows.append({
            "code": code,
            "language": name,
            "family": family,
            "tokens": total_tokens,
            "coverage": transcribed_tokens / total_tokens,
            "triphones": real["triphones"],
            "peak_d1": real["peak_d1"],
            "peak_d2": real["peak_d2"],
            "peak_pct": real["peak_pct"],
            "mean_d_real": real["mean_d"],
            "mean_d_shuf": shuf["mean_d"],
            "std_d_real": real["std_d"],
            "std_d_shuf": shuf["std_d"],
            "H_d1d2_real": real["entropy_d1_d2"],
            "H_d1d2_shuf": shuf["entropy_d1_d2"],
            "mean_d1": real["mean_d1"],
            "mean_d2": real["mean_d2"],
            "frac_d1_gt_d2": real["frac_d1_gt_d2"],
            "wilcoxon_p": real["wilcoxon_p"],
            "word_r_real": real["word_r"],
            "word_r_shuf": shuf["word_r"],
        })
        heatmaps.append((name, real["mat_prob"], shuf["mat_prob"]))

        print(
            f"{name:10s} | peak=({real['peak_d1']:2d},{real['peak_d2']:2d}) | "
            f"E[d] real={real['mean_d']:.2f} vs shuf={shuf['mean_d']:.2f} | "
            f"std[d] real={real['std_d']:.2f} vs shuf={shuf['std_d']:.2f} | "
            f"H real={real['entropy_d1_d2']:.2f} vs shuf={shuf['entropy_d1_d2']:.2f} | "
            f"r real={real['word_r']:.3f} vs shuf={shuf['word_r']:.3f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/summary.csv", index=False)

    # Figure 1: Heatmaps of P(d1, d2) for Real texts
    n_langs = len(heatmaps)
    ncols = 4
    nrows = int(np.ceil(n_langs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), constrained_layout=True)
    axes_flat = axes.flatten()
    for idx, (name, mat_real, _) in enumerate(heatmaps):
        ax = axes_flat[idx]
        im = ax.imshow(mat_real[:18, :18] * 100, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("$d_2$ (2nd–3rd phone)")
        ax.set_ylabel("$d_1$ (1st–2nd phone)")
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_yticks([0, 4, 8, 12, 16])
    for idx in range(n_langs, len(axes_flat)):
        axes_flat[idx].axis("off")
    fig.colorbar(im, ax=axes_flat[:n_langs], label="Probability (%)", shrink=0.6)
    fig.savefig("figures/triphone_heatmaps.pdf")
    plt.close(fig)

    # Figure 2: Real vs Shuffled comparison (Mean distance, Std distance, Joint Entropy, Word r)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.38

    panels = [
        (axes[0, 0], "mean_d_real", "mean_d_shuf", "Mean Adjacent Feature Distance $\\mathbb{E}[d]$", "Distance (features)"),
        (axes[0, 1], "std_d_real", "std_d_shuf", "Std. Dev. of Adjacent Distance $\\sigma(d)$", "Std. Dev. (features)"),
        (axes[1, 0], "H_d1d2_real", "H_d1d2_shuf", "Joint Entropy $H(d_1, d_2)$", "Entropy (bits)"),
        (axes[1, 1], "word_r_real", "word_r_shuf", "Word-Level Mean–Std Correlation $r$", "Pearson $r$"),
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

    fig.savefig("figures/real_vs_shuffled_comparison.pdf")
    plt.close(fig)
    print("\nSaved results/summary.csv and figures/*.pdf")


if __name__ == "__main__":
    main()
