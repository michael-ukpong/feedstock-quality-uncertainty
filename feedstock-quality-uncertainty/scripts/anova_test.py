"""
Reproduces the one-way ANOVA reported in Section 4.1 of the study:
tests whether the six feedstock classes differ significantly in composition
and in the derived ethanol proxy (CTEP).

Usage: python anova_test.py
Requires: pandas, scipy
Input: analysis_subset_with_proxy.csv (the verified 59-record subset)
"""
import pandas as pd
from scipy import stats

df = pd.read_csv("analysis_subset_with_proxy.csv")

print(f"n = {len(df)} records across {df['cls'].nunique()} feedstock classes\n")
print("One-way ANOVA: does feedstock class explain composition differences?")
print(f"{'Variable':14s} {'F':>8s} {'p-value':>10s}")
for col in ["Cellulose", "Hemicellulose", "Lignin", "Ash", "ctep"]:
    groups = [g[col].values for _, g in df.groupby("cls")]
    F, p = stats.f_oneway(*groups)
    print(f"{col:14s} {F:8.2f} {p:10.4f}")
