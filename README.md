# Cross-Linguistic Distinctive-Feature Constraints on Phonemic Triphones

This repository contains the analysis pipeline, summary datasets, figures, and LaTeX manuscript for studying how distinctive-feature transitions in word-internal triphones $(\phi_1, \phi_2, \phi_3)$ are organized across typologically diverse languages.

- **Manuscript:** [`paper.tex`](paper.tex) / [`paper.pdf`](paper.pdf) (formatted with the Taylor & Francis `interact.cls` template)

---

## 1. Project Overview

Phonological systems are shaped by competing communicative pressures—often framed in terms of articulatory economy and perceptual distinctiveness. While classical dispersion models have focused primarily on static phoneme inventories (e.g., vowel spacing), this project investigates **sequential feature-space constraints** across consecutive transcribed segments in running text.

For every word of length $L \geq 3$, we compute the feature-space distance between adjacent segments under two complementary metrics over PanPhon's 24-dimensional ternary articulatory vectors $\mathbf{v}(\phi) \in \{-1, 0, +1\}^{24}$:
1. **Feature Hamming distance** ($d_{\text{H}} \in \{0, \dots, 24\}$): count of non-shared feature values.
2. **Rescaled cosine distance** ($d_{\cos} \in [0, 1]$): $\ell_2$-normalized angular separation, $\frac{1}{2}(1 - \cos(\mathbf{v}_a, \mathbf{v}_b))$, which controls for varying numbers of unspecified (`0`) entries across segment classes.

To isolate genuine sequential phonotactic constraints from marginal phoneme frequencies, word-length distributions, and coarse syllable alternation, natural words are compared against **two nested permutation baselines**:
- **Unrestricted within-word shuffle:** randomly permutes segments within each word while preserving its exact segment multiset and corpus token count.
- **C/V-template-preserving baseline:** preserves every word's exact vocalic/non-vocalic (`syl == +1` vs. `syl != +1`) template while permuting consonants across C slots and vowels across V slots in the corpus.

---

## 2. Data and Languages

Word forms and corpus token frequencies are extracted from the training splits of **13 Universal Dependencies (UD v2) treebanks** spanning five language families (excluding punctuation `UPOS == PUNCT`, multiword token ranges, and non-alphabetic tokens):

| Code | Language | Family | UD v2 Treebank | G2P / Lexicon | Token Coverage |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `en` | English | Indo-European (Germanic) | `UD_English-EWT` | `cmudict` v1.1.3 | 97.0% |
| `de` | German | Indo-European (Germanic) | `UD_German-GSD` | `epitran` (`deu-Latn`) | 99.98% |
| `pt` | Portuguese | Indo-European (Romance) | `UD_Portuguese-Bosque` | `epitran` (`por-Latn`) | 99.86% |
| `es` | Spanish | Indo-European (Romance) | `UD_Spanish-GSD` | `epitran` (`spa-Latn`) | 99.86% |
| `fr` | French | Indo-European (Romance) | `UD_French-GSD` | `epitran` (`fra-Latn`) | 99.95% |
| `pl` | Polish | Indo-European (Slavic) | `UD_Polish-PDB` | `epitran` (`pol-Latn`) | 100.0% |
| `ru` | Russian | Indo-European (Slavic) | `UD_Russian-GSD` | `epitran` (`rus-Cyrl`) | 99.98% |
| `hi` | Hindi | Indo-European (Indo-Aryan) | `UD_Hindi-HDTB` | `epitran` (`hin-Deva`) | 100.0% |
| `hu` | Hungarian | Uralic | `UD_Hungarian-Szeged` | `epitran` (`hun-Latn`) | 100.0% |
| `fi` | Finnish | Uralic | `UD_Finnish-TDT` | `epitran` (`fin-Latn`) | 99.97% |
| `tr` | Turkish | Turkic | `UD_Turkish-IMST` | `epitran` (`tur-Latn`) | 100.0% |
| `id` | Indonesian | Austronesian | `UD_Indonesian-GSD` | `epitran` (`ind-Latn`) | 99.93% |
| `ko` | Korean | Koreanic | `UD_Korean-GSD` | `epitran` (`kor-Hang`) | 99.67% |

*(Raw `.conllu` files are excluded from git via `.gitignore` and can be downloaded from [universaldependencies.org](https://universaldependencies.org/).)*

---

## 3. Main Findings

1. **Universal intermediate-distance mode:** Across all 13 languages, the token-weighted joint triphone distance distribution $P(d_1, d_2)$ peaks at intermediate feature distances—$(d_1^*, d_2^*) \in \{(8,8), (9,9), (9,10), (10,10)\}$ under Hamming distance and $[0.28, 0.38]$ under rescaled cosine distance.
2. **Consistent real vs. unrestricted-shuffle differences (13/13 languages):** Relative to unrestricted within-word permutations, natural text exhibits higher expected adjacent distance ($\mathbb{E}[d]$), lower standard deviation ($\sigma(d)$), and lower joint triphone entropy ($H(d_1, d_2)$) in all 13 languages under both metrics (paired Wilcoxon $W=0, p = 2.44 \times 10^{-4}$). These results hold across 20 Monte Carlo shuffle seeds, 17k-token subsampling, and type-weighted analyses.
3. **Separating C/V alternation from sub-syllabic sequencing:** Under the C/V-template-preserving baseline, the upward shift in $\mathbb{E}[d]$ disappears ($6/13$ languages, $p = 0.946$), showing it is driven by C/V alternation. However, natural sequences still show significantly narrower dispersion $\sigma(d)$ ($11/13$ languages, $p = 4.64 \times 10^{-3}$) and lower joint entropy $H(d_1, d_2)$ ($13/13$ languages, $W=0, p = 2.44 \times 10^{-4}$), revealing sub-syllabic feature-sequencing constraints beyond C/V alternation.
4. **Negative replications of single-language patterns:** The negative within-word mean–SD correlation ($r$) appears similarly in shuffled baselines (indicating a combinatorial property of finite discrete sequences), and directional triphone asymmetry ($d_1 > d_2$) does not hold cross-linguistically after multiple-testing correction.

---

## 4. Repository Structure & Reproduction

- `analyze_crosslinguistic.py` — Main Hamming distance analysis (`results/summary.csv`, `figures/triphone_heatmaps.pdf`, `figures/real_vs_shuffled_comparison.pdf`).
- `analyze_cosine.py` — Rescaled cosine distance analysis (`results/summary_cosine.csv`, `figures/triphone_heatmaps_cosine.pdf`, `figures/real_vs_shuffled_cosine.pdf`).
- `analyze_cv_baseline.py` — C/V-template-preserving baseline comparison (`results/cv_baseline.csv`).
- `analyze_peak_sharpness.py` — Exploratory analysis of peak sharpness vs. inventory size and CVC/VCV alternation.
- `run_robustness_checks.py` — 20-seed Monte Carlo shuffle (`results/robustness_mc20.csv`), 17k-token subsampling (`results/robustness_sub17k.csv`), and type-weighted (`results/robustness_type.csv`) checks.
- `run_feature_sensitivity.py` — PanPhon salience-weighted, partial Hamming, 16-feature non-major-class ablation, cosine binning (10/20/40), IPA segmentation audit (`results/segmentation_audit.csv`), and 5-family equal-weight aggregation (`results/feature_sensitivity.csv`).

### Setup & Run

```bash
python3 -m venv .venv
.venv/bin/pip install panphon==0.22.2 epitran==1.35.3 cmudict==1.1.3 numpy==2.5.5 scipy==1.18.1 pandas==3.1.1 matplotlib==3.11.2
```

Place the 13 UD v2 training treebanks in `data/ud/{code}.conllu`, then run:

```bash
.venv/bin/python analyze_crosslinguistic.py
.venv/bin/python analyze_cosine.py
.venv/bin/python analyze_peak_sharpness.py
.venv/bin/python analyze_cv_baseline.py
.venv/bin/python run_robustness_checks.py

pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
