"""
Report Visualizations & Empirical Model Evaluation Analytics.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. EMPIRICAL MODEL VALIDATION & SCIENTIFIC ARTIFACTS:
   Transforms raw telemetry and trained scikit-learn models into publication-grade
   visual evidence for algorithmic performance and unsupervised clustering validity.

2. ELBOW METHOD CURVATURE ANALYSIS:
   Plots Within-Cluster Sum of Squares (WCSS Inertia) against cluster cardinality k:
       J(k) = sum_{j=1}^k sum_{x in S_j} ||x - mu_j||^2
   Visualizes the inflection point ("elbow") where marginal reduction in distortion
   diminishes, justifying the selection of k = 3 behavioral archetypes.

3. PRINCIPAL COMPONENT ANALYSIS (PCA) DIMENSIONALITY REDUCTION:
   High-dimensional normalized feature vectors in R^D (efficiency, backtracks,
   decision time, variance) are projected onto the top 2 orthogonal eigenvectors
   of the empirical covariance matrix Sigma:
       Sigma v_i = lambda_i v_i
   Visualizes cluster separation and topological playstyle boundaries in R^2.

4. EMPIRICAL SEARCH RUNTIME BENCHMARKING:
   Empirically verifies the theoretical A* computational complexity O((V + E) log V)
   across expanding grid dimensions (N x N from 6x6 to 24x24), plotting mean solve
   latency and standard deviation error bars.
===============================================================================
"""

from __future__ import annotations

import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data.preprocess import build_training_frame
from game.astar import astar_path
from game.maze_gen import generate_maze

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
DEFAULT_KMEANS_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "models", "kmeans_strategy.joblib")


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_elbow_curve(kmeans_model_path: str = DEFAULT_KMEANS_PATH):
    _ensure_dir()
    bundle = joblib.load(kmeans_model_path)
    elbow = bundle["elbow"]
    ks = sorted(elbow.keys())
    inertias = [elbow[k] for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, inertias, marker="o", color="#2563eb")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Inertia")
    ax.set_title("K-Means elbow curve — justifying K")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "elbow_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pca_clusters(kmeans_model_path: str = DEFAULT_KMEANS_PATH):
    _ensure_dir()
    bundle = joblib.load(kmeans_model_path)
    km, pca, features, cluster_names = bundle["model"], bundle["pca"], bundle["features"], bundle["cluster_names"]

    df, _ = build_training_frame()
    X = df[features].values
    coords = pca.transform(X)
    labels = km.predict(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    palette = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed"]
    for cid in sorted(set(labels)):
        mask = labels == cid
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7,
                   color=palette[cid % len(palette)],
                   label=cluster_names.get(cid, f"cluster {cid}"))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Playstyle clusters (PCA projection)")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "pca_clusters.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_astar_benchmarks(sizes=(6, 8, 10, 12, 16, 20, 24), trials=8):
    """Benchmark A* solve time vs maze size — the search-cost evidence
    figure referenced on slide 10 ('CSP build, visualised' pairs with
    this for the algorithms half of the report)."""
    _ensure_dir()
    results = {}
    for size in sizes:
        times = []
        for t in range(trials):
            maze = generate_maze(size, size, seed=t)
            start = time.perf_counter()
            astar_path(maze, maze.start, maze.exit)
            times.append(time.perf_counter() - start)
        results[size] = (np.mean(times) * 1000, np.std(times) * 1000)  # ms

    fig, ax = plt.subplots(figsize=(6, 4))
    sizes_list = list(results.keys())
    means = [results[s][0] for s in sizes_list]
    stds = [results[s][1] for s in sizes_list]
    ax.errorbar(sizes_list, means, yerr=stds, marker="o", color="#dc2626", capsize=4)
    ax.set_xlabel("Maze size (N x N)")
    ax.set_ylabel("A* solve time (ms)")
    ax.set_title("A* benchmark: solve time vs. maze size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "astar_benchmark.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_skill_breakdown():
    _ensure_dir()
    df, _ = build_training_frame()
    counts = df["skill_label"].value_counts().reindex(["Beginner", "Intermediate", "Expert"]).fillna(0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(counts.index, counts.values, color=["#93c5fd", "#3b82f6", "#1d4ed8"])
    ax.set_ylabel("Rounds")
    ax.set_title("Logged rounds by KNN skill class")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "skill_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all():
    paths = []
    paths.append(plot_elbow_curve())
    paths.append(plot_pca_clusters())
    paths.append(plot_astar_benchmarks())
    paths.append(plot_skill_breakdown())
    return paths


if __name__ == "__main__":
    for p in generate_all():
        print(f"wrote {p}")
