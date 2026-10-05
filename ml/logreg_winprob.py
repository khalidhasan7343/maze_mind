"""Train and apply an escape-probability model using mid-round snapshots. Grouped round
splits evaluate the estimate without placing related snapshots on both sides of the
evaluation boundary.

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
   vectors in the classic HUD. The first prediction loads its model artifact;
   subsequent predictions reuse the cache instead of repeating disk access.
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
from data.simulate import PROFILES, iter_bot_round
from ml.artifacts import metadata, validate_bundle
from data.logger import RoundLogger

import os

FEATURE_COLS = ["dist_to_hunter", "dist_to_exit", "time_left_ratio", "efficiency"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "logreg_winprob.joblib")

_MODEL_CACHE: dict | None = None


def collect_snapshots(n_rounds_per_profile: int = 60, snapshots_per_round: int = 6,
                       seed: int = 1) -> pd.DataFrame:
    """Sample running bot rounds and attach their eventual outcomes so the model learns to
    estimate escape probability from an unfinished game state.

    Run bot rounds, but instead of only logging the final outcome,
    sample mid-round feature snapshots and label every snapshot with the
    round's *eventual* outcome — that's the supervised signal for
    'given how things look right now, did this player go on to escape?'
    """
    rng = random.Random(seed)
    rows = []
    if n_rounds_per_profile < 1 or snapshots_per_round < 1:
        raise ValueError("Round and snapshot counts must be positive")

    for profile in PROFILES:
        for round_index in range(n_rounds_per_profile):
            cfg = RoundConfig(
                width=rng.choice([8, 10, 12]),
                height=rng.choice([8, 10, 12]),
                round_time_limit_s=45.0,
                hunter_replan_interval=rng.choice([10, 15, 20]),
                hunter_move_interval=rng.choice([15, 20, 25]),
                seed=rng.randint(0, 1_000_000),
            )
            snapshots = []
            interval = cfg.round_time_limit_s / snapshots_per_round
            next_snapshot = 0.0
            for engine in iter_bot_round(profile, cfg, rng):
                if engine.outcome == Outcome.PLAYING and engine.elapsed_s + 1e-10 >= next_snapshot:
                    snapshots.append(dict(engine.win_probability_features()))
                    next_snapshot += interval

            won = 1 if engine.outcome == Outcome.ESCAPED else 0
            for snap in snapshots:
                snap["won"] = won
                snap["round_id"] = f"{seed}-{profile.name}-{round_index}"
                snap["is_bot"] = 1
                rows.append(snap)

    return pd.DataFrame(rows)


def split_snapshot_rounds(df: pd.DataFrame):
    """Keep every snapshot from a round in one evaluation partition so related game states
    cannot leak from training into testing.

    Stratify whole rounds, never placing snapshots of one round in both sets.
    """
    rounds = df.groupby("round_id")["won"].first()
    counts = rounds.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("Win probability needs at least two escaped and two unsuccessful rounds")
    train_ids, test_ids = train_test_split(rounds.index.to_numpy(), test_size=max(2, int(np.ceil(len(rounds) * 0.2))),
                                         random_state=42, stratify=rounds.to_numpy())
    return df[df["round_id"].isin(train_ids)], df[df["round_id"].isin(test_ids)]


def train(model_path: str = MODEL_PATH, snapshot_frame: pd.DataFrame | None = None, **collect_kwargs):
    """Evaluate escape prediction on separate rounds, then save a refitted model so the HUD
    can use all eligible examples after evaluation is complete.
    """
    global _MODEL_CACHE
    df = collect_snapshots(**collect_kwargs) if snapshot_frame is None else snapshot_frame.copy()
    if snapshot_frame is None:
        human = pd.DataFrame(RoundLogger().fetch_snapshots())
        if not human.empty:
            df = pd.concat([df, human[human["is_bot"] == 0]], ignore_index=True)
    if df.empty:
        raise ValueError("No snapshots available for training")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + ["won", "round_id", "is_bot"])

    X = df[FEATURE_COLS].values
    y = df["won"].values

    training, testing = split_snapshot_rounds(df)
    X_train, X_test = training[FEATURE_COLS].values, testing[FEATURE_COLS].values
    y_train, y_test = training["won"].values, testing["won"].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    metrics = {"auc": float(roc_auc_score(y_test, proba)), "accuracy": float(accuracy_score(y_test, pred)),
               "train_rounds": int(training["round_id"].nunique()), "test_rounds": int(testing["round_id"].nunique())}
    print(f"Logistic regression AUC: {metrics['auc']:.3f}")
    print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
    print("Coefficients:", dict(zip(FEATURE_COLS, clf.coef_[0].round(3))))
    for is_bot, domain in [(1, "synthetic"), (0, "human")]:
        mask = testing["is_bot"].to_numpy() == is_bot
        metrics[domain] = {"samples": int(mask.sum()), "accuracy": float(accuracy_score(y_test[mask], pred[mask])) if mask.any() else None,
                           "auc": float(roc_auc_score(y_test[mask], proba[mask])) if len(set(y_test[mask])) == 2 else None}
        print(f"{domain} held-out evaluation: {metrics[domain]}")
    clf.fit(X, y)

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    bundle = {"model": clf, "features": FEATURE_COLS, "metrics": metrics,
              "_path": model_path, **metadata()}
    joblib.dump(bundle, model_path)
    _MODEL_CACHE = bundle
    print(f"Saved model -> {model_path}")
    return clf


def predict_win_probability(features: dict, model_path: str = MODEL_PATH) -> float:
    """Run cached inference on current engine features so the classic HUD can refresh its
    escape estimate without repeatedly loading the model from disk.

    Called once per frame during play — must be cheap. features come
    straight from RoundEngine.win_probability_features(), unscaled (the
    model was trained on raw values, so no scale_params needed here).
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None or _MODEL_CACHE.get("_path") != model_path:
        bundle = validate_bundle(joblib.load(model_path))
        bundle["_path"] = model_path
        _MODEL_CACHE = bundle
    else:
        bundle = _MODEL_CACHE

    clf = bundle["model"]
    x = np.array([[features[c] for c in FEATURE_COLS]])
    return float(clf.predict_proba(x)[0, 1])


if __name__ == "__main__":
    train()
