"""Train and apply a KNN skill classifier and map recent skill history to hunter cadence.
Independent labels and split-specific scaling make the reported evaluation separate from
automatic difficulty predictions.

KNN Skill Classifier & Dynamic Difficulty Adaptation.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. SUPERVISED CLASSIFICATION:
   K-Nearest Neighbors (KNN) is a non-parametric, instance-based lazy learning
   algorithm. Given a training dataset D = {(x_i, y_i)}, KNN classifies an
   unlabelled query instance x_q by identifying its k nearest neighbors in the
   feature space and performing majority voting:
       y_pred = argmax_{c in C} sum_{i in N_k(x_q)} I(y_i == c)

2. DISTANCE METRIC & THE MANDATE OF NORMALIZATION:
   Distance between feature vectors u, v is measured via L2 Euclidean metric:
       d(u, v) = sqrt( sum_{j=1}^D (u_j - v_j)^2 )
   CRITICAL NOTE: Distance-based models are intrinsically sensitive to feature
   scale. If one feature spans [0, 60] (e.g. time_taken_s) while another spans
   [0, 1] (e.g. efficiency_ratio), the former would dominate the squared-error
   sum by a factor of 3600:1! This project uses min-max scaling to make
   feature ranges comparable before measuring distances. Each evaluation
   split fits its own scaler using only its training rows.

3. "DIFFICULTY WITHOUT CHEATING" PRINCIPLE:
   In game AI, artificial difficulty is frequently fabricated by giving the AI
   unrealistic cheats (speed hacks, teleportation, or artificially nerfing the
   player). Here, the A* algorithm is ALWAYS mathematically optimal. The only
   knobs tuned by the skill classifier are CADENCE parameters:
     - hunter_move_interval: Frame delay between physical cell steps.
     - hunter_replan_interval: Frequency of path re-evaluations.
     - awakening_duration_s: Initial grace period giving the player breathing room.
===============================================================================
"""

from __future__ import annotations

import os

import joblib
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

from data.preprocess import build_feature_frame
from ml.artifacts import metadata, validate_bundle

FEATURE_COLS = [
    "time_taken_s",
    "efficiency_ratio",
    "backtracks",
    "avg_decision_time_s",
    "decision_time_variance",
    "outcome_code",
]

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "knn_skill.joblib")

_MODEL_CACHE: dict | None = None

# Balanced Cadence Knobs per Skill Class (30 FPS base)
DIFFICULTY_BY_SKILL = {
    "Beginner": {
        "hunter_replan_interval": 30,
        "hunter_move_interval": 38,       # ~1.27s per cell: forgiving, educational pacing
        "awakening_duration_s": 3.0,      # Generous startup grace period
    },
    "Intermediate": {
        "hunter_replan_interval": 20,
        "hunter_move_interval": 28,       # ~0.93s per cell: balanced, brisk pursuit
        "awakening_duration_s": 2.2,
    },
    "Expert": {
        "hunter_replan_interval": 12,
        "hunter_move_interval": 20,       # ~0.66s per cell: relentless, demands tactical abilities
        "awakening_duration_s": 1.5,
    },
}


def train(db_path: str | None = None, k: int = 5, model_path: str = MODEL_PATH):
    """Evaluate and save a skill classifier using independently labelled rounds and split-
    specific scaling so accuracy is measured without preprocessing leakage.

    Evaluate raw telemetry with scaling fitted independently in each split.
    """
    global _MODEL_CACHE
    df = build_feature_frame(db_path)
    labelled = df[df["label_source"].isin(["synthetic", "verified"])].dropna(subset=["skill_label"])
    counts = labelled["skill_label"].value_counts()
    if len(counts) < 2 or counts.min() < 3 or len(labelled) < 10:
        raise ValueError("KNN needs at least 10 labelled rounds, two skill classes, and 3 rounds per class")

    X = labelled[FEATURE_COLS].values
    y = labelled["skill_label"].values

    train_indices, test_indices = train_test_split(
        np.arange(len(labelled)), test_size=max(len(counts), int(np.ceil(len(labelled) * 0.2))), random_state=42, stratify=y
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    folds = min(5, int(counts.min()))
    max_k = min(len(X_train), len(X) - int(np.ceil(len(X) / folds)))
    if k < 1 or k > max_k:
        raise ValueError(f"KNN k must be between 1 and {max_k} for the evaluation splits")
    clf = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42), error_score="raise")

    print(f"KNN (k={k}) held-out accuracy: {acc:.3f}")
    print(f"{folds}-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    metrics = {"held_out_accuracy": float(acc), "cv_accuracy": float(cv_scores.mean())}
    for is_bot, domain in [(1, "synthetic"), (0, "human")]:
        mask = labelled.iloc[test_indices]["is_bot"].to_numpy() == is_bot
        metrics[domain] = {"samples": int(mask.sum()), "accuracy": float(accuracy_score(y_test[mask], y_pred[mask])) if mask.any() else None}
        print(f"{domain} held-out evaluation: {metrics[domain]}")

    # Evaluation is complete; the deployment pipeline can now use all labelled rounds.
    clf.fit(X, y)
    scaler = clf.named_steps["minmaxscaler"]
    scale_params = {c: {"min": float(lo), "max": float(hi)} for c, lo, hi in zip(FEATURE_COLS, scaler.data_min_, scaler.data_max_)}

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bundle = {"model": clf, "scale_params": scale_params, "features": FEATURE_COLS,
              "metrics": metrics, "_path": model_path, **metadata()}
    joblib.dump(bundle, model_path)
    _MODEL_CACHE = bundle
    print(f"Saved model -> {model_path}")
    return clf, scale_params


def predict_skill(feature_row: dict, model_path: str = MODEL_PATH) -> str:
    """Use a cached validated pipeline on raw round features so completed-round skill
    estimates include the scaler learned during training.

    Classify player skill based on unscaled telemetry features from a completed round.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None or _MODEL_CACHE.get("_path") != model_path:
        bundle = validate_bundle(joblib.load(model_path))
        bundle["_path"] = model_path
        _MODEL_CACHE = bundle
    else:
        bundle = _MODEL_CACHE

    clf = bundle["model"]
    x = np.array([[feature_row[c] for c in bundle["features"]]])
    return clf.predict(x)[0]


def difficulty_for_skill(skill: str) -> dict:
    """Return a copy of the skill cadence settings so adaptive rounds can adjust hunter
    timing without mutating the shared preset dictionary.

    Retrieve cadence tuning parameters corresponding to skill level.
    """
    return dict(DIFFICULTY_BY_SKILL.get(skill, DIFFICULTY_BY_SKILL["Intermediate"]))


def skill_from_history(rows: list, window: int = 5, default: str = "Beginner") -> str:
    """Reduce recent skill labels to a median tier so one unusual result does not abruptly
    change the next round difficulty.

    Median recent tier prevents a single unusual round from changing difficulty.
    """
    tiers = ["Beginner", "Intermediate", "Expert"]
    labels = [r.get("predicted_skill") or r.get("skill_label") for r in rows[-window:]]
    ranks = [tiers.index(label) for label in labels if label in tiers]
    return tiers[int(np.median(ranks))] if ranks else default


if __name__ == "__main__":
    train()
