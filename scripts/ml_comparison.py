"""
Honest, non-circular ML comparison for the dissertation's Methodology/Results chapters.
Question: given ONLY the crop-class label (one-hot encoded) as input, do RF/GBM/XGBoost/SVR
add predictive value over a simple linear-regression-on-dummies (equivalent to class means),
under the SAME leave-one-source-out validation used throughout this project?
This is NOT circular: class label is not a deterministic formula input to CTEP (unlike
cellulose/hemicellulose, which by definition would give R^2=1 trivially and must never be
presented as an ML result).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

df = pd.read_csv("analysis_subset_with_proxy.csv")
y = df["ctep"].values
cls = pd.factorize(df["cls"])[0]
src = pd.factorize(df["source"])[0]
X_dummies = pd.get_dummies(df["cls"]).values.astype(float)  # one-hot, NOT a formula input to CTEP

def models(seed=42):
    return {
        "Linear Regression": LinearRegression(),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed, verbosity=0),
    }

def loso_predict(model, X, y, groups):
    preds = np.empty(len(y))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() < 2:
            preds[te] = y[tr].mean() if tr.sum() else y.mean()
            continue
        m = model
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return preds

print(f"n={len(y)}, sources={len(np.unique(src))}, classes={len(np.unique(cls))}\n")
print("Leave-one-source-out performance using ONLY crop-class (one-hot) as input:")
print(f"{'Model':20s} {'RMSE':>8s} {'R2':>8s}")
baseline_rmse = None
for name, m in models().items():
    preds = loso_predict(m, X_dummies, y, src)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    ss_res = np.sum((y - preds) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    if name == "Linear Regression":
        baseline_rmse = rmse
    print(f"{name:20s} {rmse:8.2f} {r2:8.3f}")

print(f"\n(For reference, RMSE from the simple class-mean/pooled-mean approach reported")
print(f"elsewhere in this study: class-mean RMSE = 33.7, pooled-mean RMSE = 36.9 kg/Mg)")
