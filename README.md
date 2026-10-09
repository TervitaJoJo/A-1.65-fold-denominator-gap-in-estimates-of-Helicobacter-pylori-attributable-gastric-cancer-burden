# External denominator sensitivity analysis

This anonymous repository contains the code for the figure-based external-incidence-denominator sensitivity analysis accompanying the correspondence on *Helicobacter pylori*-attributable gastric-cancer burden in China.

The analysis compares age-specific 2022 stomach-cancer incidence from GBD 2023 with an NCC–IARC/GLOBOCAN-aligned series. It approximates age-specific infection prevalence from the published posterior-median surface in Figure 4b of the target article, applies the reported 10-year lag and Levin population-attributable fraction formula, and reports deterministic denominator scenarios.

The script does not reproduce the target article’s Bayesian model, recover posterior draws, or provide a probabilistic uncertainty interval. It should be described as a figure-based approximation and external denominator scenario.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_analysis.py
├── data/
│   └── README.md
└── results/
    └── README.md
```

No source PDF, raw download, personal information, local path, or author-identifying metadata is required in this repository. Place data locally as described in `data/README.md` or provide paths directly on the command line.

## Installation and use

```bash
python -m pip install -r requirements.txt
python run_analysis.py \
  --gbd data/raw/gbd_stomach_incidence_2022_both.csv \
  --globocan data/raw/globocan_stomach_age_groups_2022.csv \
  --figure-pdf data/raw/target_article.pdf \
  --output results/run_2022
```

The default PDF page is 9 (the page containing Figure 4b in the supplied article PDF). Use `--page` if a different PDF pagination is used.

Outputs include primary reconstructed counts, age-specific intermediate values, the complete deterministic sensitivity table, contour checks, and a machine-readable audit JSON.

## Interpretation

The primary point estimates are conditional on the published Figure 4b surface, the reported relative risks, the 10-year lag, and the stated treatment of the GLOBOCAN 75+ residual. The deterministic range reflects extraction and assumption checks; it is not a probability interval and does not quantify uncertainty in the unreleased numerical surface.

## Citation and data provenance

The code is intended to accompany the final correspondence and its supplementary appendix. GBD Results Tool and GLOBOCAN API citation details, access dates, and input definitions should be preserved in the manuscript’s references and data-availability statement.
