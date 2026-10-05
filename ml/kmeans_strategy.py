"""Train and apply playstyle clustering with saved scaling, names, and PCA. Keeping these
outputs in one artifact makes live style names and report coordinates refer to the same
learned groups.

K-Means Playstyle Clustering & PCA Dimensionality Reduction.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. UNSUPERVISED CLASSIFICATION (CLUSTERING):
   K-Means partitions N unlabeled player observations into K distinct clusters,
   grouping playthroughs based purely on geometrical proximity in feature space:
   e.g. "cautious explorers" vs. "aggressive rushers".

2. OBJECTIVE FUNCTION & LLOYD'S ALGORITHM:
   K-Means minimizes the Within-Cluster Sum of Squares (WCSS), termed Inertia:
       J = sum_{k=1}^K sum_{x in S_k} ||x - mu_k||^2
   Iterative optimization cycles two steps:
     - Assignment: Assign each x_i to closest centroid mu_k:
           c_i = argmin_k ||x_i - mu_k||^2
     - Update: Recompute centroid as mean of cluster members:
           mu_k = (1 / |S_k|) * sum_{x in S_k} x

3. MODEL SELECTION (THE ELBOW METHOD):
   As K approaches N, inertia monotonically decreases to 0. The optimal K is
   chosen at the "elbow" point where marginal reductions in inertia exhibit
   diminishing returns, balancing parsimony with cluster cohesion.

4. DIMENSIONALITY REDUCTION (PRINCIPAL COMPONENT ANALYSIS - PCA):
   PCA projects the 5D feature space into an orthogonal 2D eigenspace maximizing
   explained variance:
       Sigma = (1 / N) * X^T X = V Lambda V^T
   The top 2 eigenvectors (PC1, PC2) capture the primary axes of playstyle variance,
   enabling 2D visualization in dashboard/report_plots.py.
===============================================================================
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from data.preprocess import build_training_frame
from ml.artifacts import metadata, validate_bundle

FEATURE_COLS = [
    "time_taken_s",
    "efficiency_ratio",
    "backtracks",
    "avg_decision_time_s",
    "decision_time_variance",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "kmeans_strategy.joblib")

_MODEL_CACHE: dict | None = None


def elbow_scan(X: np.ndarray, k_range=range(2, 8)) -> dict:
    """Measure clustering inertia across candidate group counts so the dashboard can show
    evidence for the selected number of playstyles.

    Inertia per K — used to justify the K choice in the report, and
    to draw the elbow curve in the dashboard.
    """
    inertias = {}
    for k in k_range:
        if k > len(X):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        inertias[k] = km.inertia_
    return inertias


def _label_clusters(df, labels) -> dict:
    """Assign interpretable names from cluster behavior so players and reports see
    descriptive playstyles instead of arbitrary numeric IDs.

    Heuristic naming: rank clusters by mean efficiency_ratio and
    avg_decision_time_s so the names mean something in the report,
    rather than just 'cluster 0/1/2'.
    """
    import pandas as pd
    tmp = df.copy()
    tmp["cluster"] = labels
    stats = tmp.groupby("cluster")[["efficiency_ratio", "avg_decision_time_s", "backtracks"]].mean()
    names = {}
    used = set()
    fast_efficient = stats.sort_values("efficiency_ratio", ascending=False).index
    for rank, cluster_id in enumerate(fast_efficient):
        row = stats.loc[cluster_id]
        if row["avg_decision_time_s"] < stats["avg_decision_time_s"].median() and row["efficiency_ratio"] > stats["efficiency_ratio"].median():
            base = "aggressive rusher"
        elif row["backtracks"] > stats["backtracks"].median():
            base = "cautious explorer"
        else:
            base = f"balanced runner"
        name = base if base not in used else f"{base} ({cluster_id})"
        used.add(name)
        names[cluster_id] = name
    return names


def train(db_path: str | None = None, k: int = 3, model_path: str = MODEL_PATH):
    """Fit and save clustering, normalization, PCA, and elbow results together so inference
    and visual reports use one consistent training coordinate system.
    """
    global _MODEL_CACHE
    df, scale_params = build_training_frame(db_path)
    X = df[FEATURE_COLS].values
    if k < 2 or k > len(X) or len(X) < 2:
        raise ValueError("K-Means needs at least two rounds and 2 <= k <= sample count")

    elbow = elbow_scan(X)
    print("Elbow inertias:", {k_: round(v, 2) for k_, v in elbow.items()})

    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    cluster_names = _label_clusters(df, labels)
    print("Cluster sizes and names:")
    for cid in sorted(set(labels)):
        print(f"  cluster {cid} ({cluster_names[cid]}): {(labels == cid).sum()} rounds")

    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(X)
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bundle = {
        "model": km, "scale_params": scale_params, "features": FEATURE_COLS,
        "cluster_names": cluster_names, "pca": pca, "elbow": elbow,
        "_path": model_path, **metadata(),
    }
    joblib.dump(bundle, model_path)
    _MODEL_CACHE = bundle
    print(f"Saved model -> {model_path}")
    return km, labels, coords_2d, cluster_names


def predict_style(feature_row: dict, model_path: str = MODEL_PATH) -> str:
    """Apply the saved scale and cached clustering model so a completed round can receive
    the playstyle name learned during training.
    """
    global _MODEL_CACHE
    from data.preprocess import apply_scale

    if _MODEL_CACHE is None or _MODEL_CACHE.get("_path") != model_path:
        bundle = validate_bundle(joblib.load(model_path))
        bundle["_path"] = model_path
        _MODEL_CACHE = bundle
    else:
        bundle = _MODEL_CACHE

    km, scale_params = bundle["model"], bundle["scale_params"]
    scaled = apply_scale(feature_row, scale_params)
    x = np.array([[scaled[c] for c in FEATURE_COLS]])
    cluster_id = km.predict(x)[0]
    return bundle["cluster_names"].get(cluster_id, f"playstyle {cluster_id}")


if __name__ == "__main__":
    train()
