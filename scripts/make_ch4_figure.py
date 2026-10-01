import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

df = pd.read_csv("analysis_subset_with_proxy.csv")
y = df["ctep"].values
src = pd.factorize(df["source"])[0]
X_dummies = pd.get_dummies(df["cls"]).values.astype(float)

def models(seed=42):
    return {
        "Linear\nRegression": LinearRegression(),
        "SVR\n(RBF)": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale")),
        "Random\nForest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "Gradient\nBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=seed, verbosity=0),
    }

def loso_predict(model, X, y, groups):
    preds = np.empty(len(y))
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if tr.sum() < 2:
            preds[te] = y[tr].mean() if tr.sum() else y.mean(); continue
        model.fit(X[tr], y[tr]); preds[te] = model.predict(X[te])
    return preds

names, rmses = [], []
for name, m in models().items():
    preds = loso_predict(m, X_dummies, y, src)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    names.append(name); rmses.append(rmse)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
colors = ["#4c72b0"] * len(names)
bars = ax.bar(names, rmses, color=colors, width=0.6)
ax.axhline(36.9, color="grey", linestyle="--", linewidth=1.2, label="Pooled-mean baseline (36.9 kg/Mg)")
for b, v in zip(bars, rmses):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Leave-one-source-out RMSE (kg ethanol / dry Mg)")
ax.set_title("Model performance using crop class alone as input\n(all algorithms converge near the simple class-mean result)")
ax.set_ylim(0, 42)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("chapter4_model_comparison.png", dpi=200)
print("saved. RMSEs:", dict(zip(names, [round(r,2) for r in rmses])))
