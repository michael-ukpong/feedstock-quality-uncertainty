**Citable archive:** The permanent, citable version of this data and code is available at: https://doi.org/10.5281/zenodo.22999675

# Feedstock Quality Uncertainty in Bioethanol Supply Chains — Data and Code

This repository contains the complete, verified data and analysis code underlying the
independent research study "Quantifying Feedstock Quality Uncertainty in Bioethanol
Supply Chains: Implications for Procurement and Tonnage Planning."

Every table and figure in the study can be regenerated from the files here. All
scripts are written to be run from the repository root, so that their references
to the `data/` and `figures/` folders resolve correctly.

## Files

- **data/analysis_subset_with_proxy.csv** — the verified 59-record analytical subset (six
  feedstock classes, 51 independent primary sources), including the derived carbohydrate-based
  ethanol proxy (CTEP). Derived from the public composition database of Durán-Aranguren (2022),
  pinned to GitHub commit `e038d35b`
  (SHA-256: `b1d8fb6f1c68821e0f5844e78557097dfb24df3798fe875c923e4e0cc423c440`).

| Script | Produces | Section / Table / Figure |
|---|---|---|
| `scripts/label_uncertainty_analysis.py` | Downloads the pinned source database, builds the 59-record subset, computes the intraclass correlation (ICC), leave-one-source-out class-mean vs. pooled-mean validation, source-clustered bootstrap confidence intervals, and the sensitivity check on the CTEP formula | Section 3.6–3.8, Section 4.3, Section 4.4, Figure 4.1 |
| `scripts/anova_test.py` | One-way ANOVA testing whether feedstock classes differ significantly in composition and in CTEP | Section 4.1 |
| `scripts/ml_comparison.py` | Leave-one-source-out comparison of five supervised learning algorithms using crop class alone as the predictor | Table 4.2 |
| `scripts/make_ch4_figure.py` | Generates the bar chart comparing the five algorithms' RMSE against the pooled-mean baseline | Figure 4.2 |

- **figures/label_uncertainty_figure.png** — Figure 4.1 (pre-generated)
- **figures/chapter4_model_comparison.png** — Figure 4.2 (pre-generated)

## Reproducing the results

Clone or download this repository, preserving its folder structure, then from the
repository root:

```
pip install pandas numpy scikit-learn xgboost matplotlib scipy

python scripts/label_uncertainty_analysis.py
python scripts/anova_test.py
python scripts/ml_comparison.py
python scripts/make_ch4_figure.py
```

## Provenance and licensing

The underlying third-party composition database (Durán-Aranguren, 2022) is licensed
GPL-3.0. This repository's own code and derived data are provided for research
reproducibility; please cite both this repository and the original database if you
reuse either.

## Citation

Ukpong, M. (2026). Quantifying Feedstock Quality Uncertainty in Bioethanol Supply
Chains: Implications for Procurement and Tonnage Planning. [Independent research
study; data and code deposited on Zenodo, DOI: 10.5281/zenodo.22999675]
