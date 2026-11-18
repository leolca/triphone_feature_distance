# Cross-Linguistic Distinctive-Feature Constraints on Phonemic Triphones

Code, summary results, figures, and manuscript for the cross-linguistic analysis of adjacent distinctive-feature distances in triphones across 13 Universal Dependencies v2 languages.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install panphon==0.22.2 epitran==1.35.3 cmudict==1.1.3 numpy==2.5.5 scipy==1.18.1 pandas==3.1.1 matplotlib==3.11.2
```

## Corpus Data

Download the 13 Universal Dependencies v2 training treebanks (`en`, `de`, `pt`, `es`, `fr`, `pl`, `ru`, `hi`, `hu`, `fi`, `tr`, `id`, `ko`) into `data/ud/{code}.conllu` from https://universaldependencies.org/.

## Reproducing Results and Manuscript

```bash
.venv/bin/python analyze_crosslinguistic.py
.venv/bin/python analyze_cosine.py
.venv/bin/python analyze_peak_sharpness.py
.venv/bin/python analyze_cv_baseline.py
.venv/bin/python run_robustness_checks.py

pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
