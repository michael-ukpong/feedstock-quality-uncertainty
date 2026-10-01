#!/usr/bin/env python3
"""
How much does the crop label tell a procurer about fermentable-carbohydrate potential?
Evidence from a literature-compiled, DOI-referenced composition database.

DATA (real, public; not modified):
  Duran-Aranguren et al., "Classification of vegetable biomass: Databases, data analysis, and
  application of Machine Learning techniques" -- file Composition_database_2022_NREL.xlsx
  Durán-Aranguren, D. (2022), Zenodo, doi:10.5281/zenodo.5851251 (GPL-3.0); GitHub duran-aranguren/biomass_composition.
  The analysed file is pinned to repository commit e038d35b (see COMMIT below).
  Each row = one literature composition record (% dry weight) with a source DOI.

OUTCOME (a PROXY, not the DOE calculator and not a measured yield):
  CTEP = 10 * (cellulose*1.111 + hemicellulose*1.136) * 0.51      [kg ethanol per dry Mg]
  It treats cellulose as glucan and hemicellulose as a pentose polymer; pectin is ignored.
  A cellulose-only lower variant and a hemicellulose-as-hexose variant are reported for sensitivity.
  Biomass needed per Mg ethanol B = 1000 / CTEP  [Mg biomass / Mg ethanol].

ANALYSES
  1. Descriptives by feedstock class (classes with >= MIN_CLASS_N records).
  2. Share of variance explained by class label (one-way ICC), with a source-clustered bootstrap CI.
  3. Leave-one-SOURCE-out prediction of CTEP from the class label alone vs the pooled mean, and the
     implied error in planned biomass tonnage. Grouping by source DOI prevents records from the same
     paper informing their own prediction.

Usage: python label_uncertainty_analysis.py [path/to/Composition_database_2022_NREL.xlsx]
       (downloads the file from GitHub if no path is given)   Requires pandas numpy matplotlib openpyxl
"""
import hashlib, io, sys, zipfile, urllib.request
import numpy as np, pandas as pd

SEED, N_BOOT, MIN_CLASS_N = 20260920, 2000, 5
COMMIT = "e038d35baa16008bfa1fc96c4fcf3b746937a2d7"     # repository commit analysed (pinned for reproducibility)
URL = f"https://codeload.github.com/duran-aranguren/biomass_composition/zip/{COMMIT}"
FILE = "Composition_database_2022_NREL.xlsx"
C6, C5, EF = 180 / 162, 150 / 132, 0.51
COMP = ["Cellulose", "Hemicellulose", "Lignin", "Pectin", "Extractives", "Protein", "Ash"]

# Documented class rules (case-insensitive substring on the 'Biomass' label; first match wins)
RULES = [("Sugarcane bagasse", "sugarcane bagasse"), ("Sugarcane straw", "sugarcane straw"), ("Rice straw", "rice straw"),
         ("Corn stover", "corn stover"), ("Wheat straw", "wheat straw"), ("Barley straw", "barley straw")]


def load(path=None):
    if path:
        data = open(path, "rb").read()
    else:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=60).read()))
        data = z.read([n for n in z.namelist() if n.endswith(FILE)][0])
    print(f"Database file {FILE}: sha256 = {hashlib.sha256(data).hexdigest()}  (repository commit {COMMIT[:8]})")
    raw = pd.read_excel(io.BytesIO(data))
    raw.columns = [c.strip() for c in raw.columns]
    for c in COMP:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw["source"] = raw["Reference/Source"].astype(str).str.strip().str.lower()
    return raw


def classify(label):
    s = str(label).lower()
    for name, key in RULES:
        if key in s:
            return name
    return None


def ctep(df, variant="main"):
    if variant == "main":
        return 10 * (df.Cellulose * C6 + df.Hemicellulose * C5) * EF
    if variant == "cellulose_only":
        return 10 * df.Cellulose * C6 * EF
    if variant == "hemi_as_hexose":
        return 10 * (df.Cellulose + df.Hemicellulose) * C6 * EF
    raise ValueError(variant)


def icc_oneway(y, g):
    """ICC(1) from one-way ANOVA: (MSB - MSW) / (MSB + (k0-1) MSW); unbalanced-safe k0."""
    y = np.asarray(y, float)
    _, gi = np.unique(np.asarray(g), return_inverse=True)
    k, n = gi.max() + 1, len(y)
    ni = np.bincount(gi, minlength=k).astype(float)
    means = np.bincount(gi, weights=y, minlength=k) / ni
    gm = y.mean()
    ssb = np.sum(ni * (means - gm) ** 2)
    ssw = np.sum((y - means[gi]) ** 2)
    msb, msw = ssb / (k - 1), ssw / (n - k)
    k0 = (n - np.sum(ni ** 2) / n) / (k - 1)
    return (msb - msw) / (msb + (k0 - 1) * msw)


def loso_arrays(y, c, s):
    """Leave-one-SOURCE-out predictions of y from class mean vs pooled mean (integer codes c, s)."""
    C, S = c.max() + 1, s.max() + 1
    Ntot, Ytot = len(y), y.sum()
    Ns, Ys = np.bincount(s, minlength=S), np.bincount(s, weights=y, minlength=S)
    Nc, Yc = np.bincount(c, minlength=C), np.bincount(c, weights=y, minlength=C)
    Ncs, Ycs = np.zeros((C, S)), np.zeros((C, S))
    np.add.at(Ncs, (c, s), 1); np.add.at(Ycs, (c, s), y)
    pool = (Ytot - Ys[s]) / (Ntot - Ns[s])
    n_cls = Nc[c] - Ncs[c, s]
    sum_cls = Yc[c] - Ycs[c, s]
    pred_cls = np.where(n_cls > 0, sum_cls / np.maximum(n_cls, 1), pool)
    return pred_cls, pool


def metrics(y, c, s):
    pc, pp = loso_arrays(y, c, s)
    sse_c, sse_p = np.sum((y - pc) ** 2), np.sum((y - pp) ** 2)
    err = (1000 / y - 1000 / pc) / (1000 / pc) * 100      # true vs label-planned biomass need, %
    return dict(skill=1 - sse_c / sse_p, rmse_class=np.sqrt(sse_c / len(y)), rmse_pool=np.sqrt(sse_p / len(y)),
                mape_class=np.mean(np.abs((y - pc) / y)) * 100, err=err)


def cluster_bootstrap(y, c, s, fn, n_boot=N_BOOT, seed=SEED):
    """Resample whole sources (papers) with replacement; each drawn copy counts as its own source."""
    rng = np.random.default_rng(seed)
    src_ids = np.unique(s)
    rows = {u: np.where(s == u)[0] for u in src_ids}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(src_ids, size=len(src_ids), replace=True)
        idx = np.concatenate([rows[u] for u in pick])
        s_new = np.repeat(np.arange(len(pick)), [len(rows[u]) for u in pick])
        if len(np.unique(c[idx])) < 2:
            continue
        try:
            out.append(fn(y[idx], c[idx], s_new))
        except Exception:
            continue
    return np.array(out)


def main(path=None, make_figure=True):
    raw = load(path)
    raw["cls"] = raw["Biomass"].map(classify)
    print(f"Database rows: {len(raw)}; distinct labels: {raw['Biomass'].nunique()}; distinct primary sources: {raw['source'].nunique()}")
    sub = raw[raw.cls.notna()].copy()
    print("Labels matched by each class rule (check for contamination):")
    for cname, g in sub.groupby("cls"):
        print(f"  {cname}: {g['Biomass'].str.strip().value_counts().to_dict()}")
    counts = sub.cls.value_counts()
    keep = counts[counts >= MIN_CLASS_N].index
    sub = sub[sub.cls.isin(keep)].copy().reset_index(drop=True)
    print(f"Classes with >= {MIN_CLASS_N} records: {sub.cls.value_counts().to_dict()}; total n = {len(sub)}; sources = {sub.source.nunique()}")

    sub["ctep"] = ctep(sub)
    sub["B"] = 1000 / sub["ctep"]
    print("\n1) Descriptives by class (CTEP kg EtOH/Mg dry; B = Mg biomass per Mg ethanol)")
    t = sub.groupby("cls").agg(n=("ctep", "size"), ctep_mean=("ctep", "mean"), ctep_sd=("ctep", "std"),
                               B_p50=("B", "median"), B_p90=("B", lambda s: s.quantile(.9)),
                               ash_mean=("Ash", "mean"), lignin_mean=("Lignin", "mean")).round(2)
    t["ctep_cv_%"] = (t.ctep_sd / t.ctep_mean * 100).round(1)
    t["B_p90/p50"] = (t.B_p90 / t.B_p50).round(3)
    print(t.to_string())

    y = sub["ctep"].values
    c = pd.factorize(sub.cls)[0]
    sc = pd.factorize(sub.source)[0]
    print("\n2) Variance explained by class label (ICC(1)) with source-clustered bootstrap 95% CI")
    for name, col in [("CTEP (main proxy)", "ctep"), ("Ash", "Ash"), ("Lignin", "Lignin")]:
        v = sub[col].values
        est = icc_oneway(v, c)
        bs = cluster_bootstrap(v, c, sc, lambda yy, cc, ss: icc_oneway(yy, cc))
        print(f"  {name:18s} ICC = {est:5.2f}   95% CI [{np.percentile(bs,2.5):5.2f}, {np.percentile(bs,97.5):5.2f}]  (boot n={len(bs)})")

    print("\n3) Leave-one-source-out prediction of CTEP: class-label mean vs pooled mean")
    res = metrics(y, c, sc)
    bs = cluster_bootstrap(y, c, sc, lambda yy, cc, ss: metrics(yy, cc, ss)["skill"])
    print(f"  RMSE class-label = {res['rmse_class']:.1f} kg/Mg; pooled = {res['rmse_pool']:.1f} kg/Mg; "
          f"skill = {res['skill']:.2f}  95% CI [{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]")
    print(f"  Mean absolute % error in CTEP when planning by class label: {res['mape_class']:.1f}%")
    e = res["err"]
    print(f"  Implied error in planned biomass tonnage (true vs label-planned): median |err| = {np.median(np.abs(e)):.1f}%, "
          f"P90 |err| = {np.percentile(np.abs(e),90):.1f}%, worst shortfall (true need above plan) = {e.max():.1f}%")
    print(f"  Share of records where label-planned tonnage falls SHORT of true need by >10%: {np.mean(e > 10)*100:.0f}%")

    print("\n4) Sensitivity to the proxy definition (skill, leave-one-source-out)")
    for v in ("cellulose_only", "hemi_as_hexose"):
        yy = ctep(sub, v).values
        print(f"  {v:16s} skill = {metrics(yy, c, sc)['skill']:.2f}")

    if make_figure:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = list(t.sort_values("B_p50").index)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        rng = np.random.default_rng(1)
        for i, c in enumerate(order):
            v = sub.loc[sub.cls == c, "B"].values
            ax[0].scatter(np.full(len(v), i) + rng.uniform(-.12, .12, len(v)), v, s=22, alpha=.75)
            ax[0].hlines(np.median(v), i - .25, i + .25, color="k")
        ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels(order, rotation=30, ha="right")
        ax[0].set_ylabel("Biomass needed per Mg ethanol (Mg)"); ax[0].set_title("Spread within class (dot = one literature record)")
        ax[1].hist(e, bins=20, color="#4c72b0"); ax[1].axvline(0, color="k")
        ax[1].set_xlabel("True need vs plan by class mean (%), leave-one-source-out")
        ax[1].set_ylabel("Records"); ax[1].set_title("Tonnage error when planning by crop label")
        plt.tight_layout(); plt.savefig("label_uncertainty_figure.png", dpi=200)
        print("\nSaved label_uncertainty_figure.png")
    sub.to_csv("analysis_subset_with_proxy.csv", index=False)
    print("Saved analysis_subset_with_proxy.csv (the exact rows used, with computed proxy)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
