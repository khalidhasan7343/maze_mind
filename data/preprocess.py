"""Convert stored telemetry into model features and reusable numeric scales. Single-round
and batch helpers keep live inference, training, and dashboard projections consistent.

Telemetry Preprocessing, Feature Engineering & Affine Normalization.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. FEATURE EXTRACTION PIPELINE:
   Raw temporal and spatial events are transformed into structured feature vectors:
     - Efficiency Ratio: net progress toward the exit / steps taken in [0, 1].
       On escaped rounds this equals optimal path length / steps taken.
     - Backtrack Frequency: Count of immediate reversals (path[t] == path[t-2]).
       Indicates spatial disorientation or dead-end entrapment.
     - Latency Distribution Moments:
       * Mean decision time mu_t: Cognitive processing velocity.
       * Variance of decision time sigma_t^2: Inconsistency, hesitation spikes.
     - Categorical Outcome Encoding:
       Discrete encoding {escaped: 0, caught: 1, timeout: 2}.

2. MIN-MAX AFFINE SCALING & METRIC EQUIVALENCE:
   Both KNN classification and K-Means clustering measure distances in Euclidean
   metric space:
       d(u, v) = sqrt( sum_{i=1}^D (u_i - v_i)^2 )
   If feature j has scale [0, 100] (e.g. time_taken_s) while feature k has scale
   [0, 2] (e.g. backtracks), dimension j dominates the metric by a factor of 2500x.
   Min-Max normalization transforms each numeric feature into [0, 1]:
       x'_i = (x_i - min_i) / (max_i - min_i)
   This prevents a feature's numeric range alone from dominating the distance.
   Scaling balances ranges; it does not prove equal predictive importance.

3. DATA DEFENSE & NUMERICAL STABILITY:
   Constant features (where max_i == min_i) are mapped to 0.0 to prevent division
   by zero. Incomplete rounds (0 steps or aborted sessions) are filtered to
   prevent degenerate efficiency ratios.
===============================================================================
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data.logger import RoundLogger

RAW_NUMERIC_FEATURES = [
    "time_taken_s",
    "efficiency_ratio",
    "backtracks",
    "avg_decision_time_s",
    "maze_size",
    "dead_end_count",
]

OUTCOME_MAP = {"escaped": 0, "caught": 1, "timeout": 2}


def load_raw_dataframe(db_path: str | None = None) -> pd.DataFrame:
    """Read logger dictionaries into a DataFrame so offline training starts from the same
    persisted telemetry as player records.
    """
    logger = RoundLogger(db_path=db_path) if db_path else RoundLogger()
    rows = logger.fetch_all()
    df = pd.DataFrame(rows)
    return df


def _repair_or_drop(df: pd.DataFrame) -> pd.DataFrame:
    """Remove incomplete or invalid examples before feature calculation so training does
    not rely on undefined movement ratios or unsupported outcomes.

    Drop rounds with no movement at all (player quit before any
    logged decision) — these carry no signal and would produce
    div-by-zero efficiency ratios. Also drop rows with invalid outcomes.
    """
    before = len(df)
    df = df[df["player_path_len"] > 0].copy()
    df = df.dropna(subset=["time_taken_s", "player_path_len", "optimal_path_len"])
    df = df[df["outcome"].isin(OUTCOME_MAP.keys())].copy()
    df = df[(df["time_taken_s"] >= 0) & (df["optimal_path_len"] >= 0)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] dropped {dropped} incomplete round(s)")
    return df


def extract_round_features(row: dict) -> dict:
    """Calculate features for one completed record so live post-round classification uses
    the same definitions as batch training.

    Extract engineered features for a single round dict (e.g. from DB)
    matching the pipeline's expected format.
    """
    decision_times = row.get("decision_times_s")
    if isinstance(decision_times, str):
        try:
            times = json.loads(decision_times) if decision_times else []
        except (TypeError, json.JSONDecodeError):
            times = []
    elif isinstance(decision_times, (list, tuple)):
        times = list(decision_times)
    else:
        times = []

    path_len = max(1, row.get("player_path_len", 1))
    optimal_len = max(0, row.get("optimal_path_len", path_len))
    outcome_str = row.get("outcome", "timeout")
    remaining = row.get("remaining_path_len")
    if remaining is None or pd.isna(remaining):
        remaining = 0 if outcome_str == "escaped" else optimal_len
    progress = max(0, optimal_len - remaining)

    return {
        "time_taken_s": float(row.get("time_taken_s", 0.0)),
        "efficiency_ratio": float(min(1.0, progress / path_len)),
        "backtracks": int(row.get("backtracks", 0)),
        "avg_decision_time_s": float(row.get("avg_decision_time_s", 0.0)),
        "decision_time_variance": float(np.var(times)) if len(times) > 1 else 0.0,
        "outcome_code": int(OUTCOME_MAP.get(outcome_str, OUTCOME_MAP["timeout"])),
        "maze_size": int(row.get("maze_width", 0) * row.get("maze_height", 0)),
        "dead_end_count": int(row.get("dead_end_count", 0)),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features across stored rounds so all models receive consistent progress,
    maze size, outcome, and timing measurements.
    """
    df = _repair_or_drop(df)

    remaining = df.get("remaining_path_len", pd.Series(np.nan, index=df.index))
    remaining = remaining.fillna(df["optimal_path_len"].where(df["outcome"] != "escaped", 0))
    df["efficiency_ratio"] = ((df["optimal_path_len"] - remaining).clip(lower=0) / df["player_path_len"]).clip(upper=1)
    df["maze_size"] = df["maze_width"] * df["maze_height"]
    df["outcome_code"] = df["outcome"].map(OUTCOME_MAP)

    def _decision_var(raw_json):
        """Decode stored decision delays defensively so absent or malformed JSON does not
        prevent batch feature extraction.
        """
        try:
            times = json.loads(raw_json) if raw_json else []
        except (TypeError, json.JSONDecodeError):
            times = []
        return float(np.var(times)) if len(times) > 1 else 0.0

    df["decision_time_variance"] = df["decision_times_s"].apply(_decision_var)

    return df


def min_max_normalize(df: pd.DataFrame, columns: list) -> tuple:
    """Fit numeric ranges and return them with the scaled data so distance-based models can
    reuse exactly the same units during inference.

    Returns (normalised_df, scale_params) so the same training ranges can be
    reapplied to new records and dashboard data. K-Means uses these saved
    parameters; KNN includes its own scaler inside its prediction pipeline.
    The logistic regression model uses unscaled snapshot features.
    """
    scale_params = {}
    out = df.copy()
    for col in columns:
        lo, hi = out[col].min(), out[col].max()
        span = hi - lo if hi != lo else 1.0
        out[col] = (out[col] - lo) / span
        scale_params[col] = {"min": float(lo), "max": float(hi)}
    return out, scale_params


def apply_scale(row: dict, scale_params: dict) -> dict:
    """Reuse saved ranges for a single record so new predictions are measured in the
    training coordinate system.

    Apply a previously-fit scale to a single new feature dict (e.g.
    live during a round) — preserves unscaled features while scaling
    columns present in scale_params.
    """
    scaled = dict(row)
    for col, params in scale_params.items():
        if col in row:
            span = params["max"] - params["min"] or 1.0
            scaled[col] = (row[col] - params["min"]) / span
    return scaled


def build_feature_frame(db_path: str | None = None, current_only: bool = True) -> pd.DataFrame:
    """Build finite, eligible, unscaled telemetry so supervised callers can split their
    data before fitting a scaler and avoid evaluation leakage.

    Engineered, unscaled data. Supervised callers split before fitting scalers.
    """
    raw = load_raw_dataframe(db_path)
    if raw.empty:
        raise ValueError("No rounds logged yet — run python main.py simulate first")
    if current_only:
        raw = raw[raw["telemetry_version"] == 2]
    engineered = engineer_features(raw)
    if engineered.empty:
        raise ValueError("No valid current telemetry — generate new rounds with python main.py simulate")
    numeric_cols = RAW_NUMERIC_FEATURES + ["decision_time_variance"]
    engineered = engineered.replace([np.inf, -np.inf], np.nan).dropna(subset=numeric_cols)
    if engineered.empty:
        raise ValueError("No finite training features")
    return engineered


def scale_frame(df: pd.DataFrame, scale_params: dict) -> pd.DataFrame:
    """Apply saved model ranges to a batch so report plots and predictions remain
    consistent when new rounds change the dataset range.

    Transform a frame using the model's saved training scale.
    """
    out = df.copy()
    for col, params in scale_params.items():
        if col in out:
            out[col] = (out[col] - params["min"]) / (params["max"] - params["min"] or 1.0)
    return out


def build_training_frame(db_path: str | None = None) -> tuple:
    """Combine feature engineering and normalization for callers needing a fully scaled
    dataset, such as unsupervised clustering.

    Full pipeline: raw rows -> engineered -> normalised.
    Returns (normalized_df, scale_params).
    """
    engineered = build_feature_frame(db_path)
    encode_cols = ["outcome_code"]  # categorical, encoded not scaled
    numeric_cols = RAW_NUMERIC_FEATURES + ["decision_time_variance"]
    normalized, scale_params = min_max_normalize(engineered, numeric_cols)
    keep = numeric_cols + encode_cols + ["skill_label", "player_tag", "session_id", "is_bot", "label_source", "predicted_skill"]
    return normalized[keep], scale_params


if __name__ == "__main__":
    df, scale = build_training_frame()
    print(df.describe())
    print(f"\n{len(df)} rows ready for modelling. Skill label counts:")
    print(df["skill_label"].value_counts())
