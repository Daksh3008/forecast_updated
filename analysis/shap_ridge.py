# analysis/shap_ridge.py
import shap
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load feature matrix
df = pd.read_csv("data/feature_matrix.csv", index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()


TARGET = "log_ret"
CLOSE = "brent_Close"

# Prepare X, y
y = df[TARGET].shift(-1).dropna()
X = df.iloc[:-1].drop(columns=[TARGET])

feature_cols = X.columns
X = X.values

# Scale
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# Train ridge
model = Ridge(alpha=1.0)
model.fit(Xs, y)

# SHAP analysis
explainer = shap.Explainer(model, Xs)
shap_values = explainer(Xs)

# Compute mean |SHAP|
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

# Ranking
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap
}).sort_values(by="mean_abs_shap", ascending=False)

importance_df.to_csv("analysis/shap_feature_ranking.csv", index=False)

print("Saved SHAP rankings to analysis/shap_feature_ranking.csv")
