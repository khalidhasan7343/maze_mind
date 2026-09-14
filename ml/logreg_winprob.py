"""
Logistic Regression — Live Probabilistic Win Estimation.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. SUPERVISED PROBABILISTIC CLASSIFICATION:
   Unlike hard-margin linear classifiers (e.g. perceptron), Logistic Regression
   models the conditional probability of binary class membership P(Y = 1 | X = x):
       P(escape | x) = sigma( w^T x + b )
   where sigma(z) is the logistic sigmoid function:
       sigma(z) = 1 / (1 + e^(-z))

2. LOG-ODDS (LOGIT) FORMULATION:
   The model linearly links input features to the logarithm of the odds ratio:
       ln( P / (1 - P) ) = beta_0 + sum_{j=1}^D beta_j * x_j
   - Positive coefficient (beta_j > 0): Feature increases odds of escape.
     (e.g., efficiency and dist_to_hunter).
   - Negative coefficient (beta_j < 0): Feature decreases odds of escape.
     (e.g., dist_to_exit: further from exit means lower escape probability).

3. MAXIMUM LIKELIHOOD ESTIMATION (MLE):
   Model parameters are optimized by minimizing binary cross-entropy loss:
       L(w) = - 1/N * sum_{i=1}^N [ y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i) ]
   via L-BFGS numerical optimization with L2 weight regularization.

4. LIVE INFERENCE IN THE HUD:
   Trained on mid-round game state snapshots, the model evaluates live feature
   vectors at 30 FPS without disk I/O, updating the HUD win-probability meter.
===============================================================================
"""

from __future__ import annotations

import random

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from game.engine import RoundEngine, RoundConfig, Outcome
from data.simulate import PROFILES, _greedy_direction, _legal_directions

import os

FEATURE_COLS = ["dist_to_hunter", "dist_to_exit", "time_left_ratio", "efficiency"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "logreg_winprob.joblib")

_MODEL_CACHE: dict | None = None


def collect_snapshots(n_rounds_per_profile: int = 60, snapshots_per_round: int = 6,
                       seed: int = 1) -> pd.DataFrame:
    """Run bot rounds, but instead of only logging the final outcome,
    sample mid-round feature snapshots and label every snapshot with the
    round's *eventual* outcome — that's the supervised signal for
    'given how things look right now, did this player go on to escape?'
    """
    rng = random.Random(seed)
    rows = []

    for profile in PROFILES:
        for _ in range(n_rounds_per_profile):
            cfg = RoundConfig(
                width=rng.choice([8, 10, 12]),
                height=rng.choice([8, 10, 12]),
                round_time_limit_s=45.0,
                hunter_replan_interval=rng.choice([10, 15, 20]),
                hunter_move_interval=rng.choice([15, 20, 25]),
                seed=rng.randint(0, 1_000_000),
            )
            engine = RoundEngine(cfg)
            snapshots = []
            max_ticks = int(cfg.round_time_limit_s * cfg.fps)
            dt = 1.0 / cfg.fps

            for tick in range(max_ticks):
                if engine.outcome != Outcome.PLAYING:
                    break
                legal = _legal_directions(engine)
                if not legal:
                    break
                if rng.random() < profile.mistake_rate:
                    direction = rng.choice(legal)
                else:
                    greedy = _greedy_direction(engine)
                    direction = greedy if (greedy and rng.random() < profile.greediness) else rng.choice(legal)
                engine.move_player(direction)
                engine.update(simulated_dt=dt)

                if tick % max(1, max_ticks // snapshots_per_round) == 0:
                    snapshots.append(dict(engine.win_probability_features()))

            won = 1 if engine.outcome == Outcome.ESCAPED else 0
            for snap in snapshots:
                snap["won"] = won
                rows.append(snap)

    return pd.DataFrame(rows)


def train(model_path: str = MODEL_PATH, **collect_kwargs):
    global _MODEL_CACHE
    df = collect_snapshots(**collect_kwargs)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df[FEATURE_COLS].values
    y = df["won"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    print(f"Logistic regression AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Coefficients:", dict(zip(FEATURE_COLS, clf.coef_[0].round(3))))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    bundle = {"model": clf, "features": FEATURE_COLS}
    joblib.dump(bundle, model_path)
    _MODEL_CACHE = bundle
    print(f"Saved model -> {model_path}")
    return clf


def predict_win_probability(features: dict, model_path: str = MODEL_PATH) -> float:
    """Called once per frame during play — must be cheap. features come
    straight from RoundEngine.win_probability_features(), unscaled (the
    model was trained on raw values, so no scale_params needed here)."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None or getattr(_MODEL_CACHE, "_path", None) != model_path:
        bundle = joblib.load(model_path)
        bundle["_path"] = model_path
        _MODEL_CACHE = bundle
    else:
        bundle = _MODEL_CACHE

    clf = bundle["model"]
    x = np.array([[features[c] for c in FEATURE_COLS]])
    return float(clf.predict_proba(x)[0, 1])


if __name__ == "__main__":
    train()
