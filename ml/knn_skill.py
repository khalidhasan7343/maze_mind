"""
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
   sum by a factor of 3600:1! Min-max normalization [0, 1] is therefore
   mathematically mandatory.

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
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.preprocess import build_training_frame

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
    """Train KNN classifier on normalized telemetry logs with 5-fold cross-validation."""
    global _MODEL_CACHE
    df, scale_params = build_training_frame(db_path)
    labelled = df.dropna(subset=["skill_label"])

    X = labelled[FEATURE_COLS].values
    y = labelled["skill_label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X, y, cv=5)

    print(f"KNN (k={k}) held-out accuracy: {acc:.3f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(classification_report(y_test, y_pred))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {"model": clf, "scale_params": scale_params, "features": FEATURE_COLS}
    joblib.dump(bundle, model_path)
    _MODEL_CACHE = bundle
    print(f"Saved model -> {model_path}")
    return clf, scale_params


def predict_skill(feature_row: dict, model_path: str = MODEL_PATH) -> str:
    """Classify player skill based on unscaled telemetry features from a completed round."""
    global _MODEL_CACHE
    from data.preprocess import apply_scale

    if _MODEL_CACHE is None or getattr(_MODEL_CACHE, "_path", None) != model_path:
        bundle = joblib.load(model_path)
        bundle["_path"] = model_path
        _MODEL_CACHE = bundle
    else:
        bundle = _MODEL_CACHE

    clf, scale_params = bundle["model"], bundle["scale_params"]
    scaled = apply_scale(feature_row, scale_params)
    x = np.array([[scaled[c] for c in FEATURE_COLS]])
    return clf.predict(x)[0]


def difficulty_for_skill(skill: str) -> dict:
    """Retrieve cadence tuning parameters corresponding to skill level."""
    return DIFFICULTY_BY_SKILL.get(skill, DIFFICULTY_BY_SKILL["Intermediate"])


if __name__ == "__main__":
    train()
