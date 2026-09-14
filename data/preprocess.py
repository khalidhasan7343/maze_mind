"""
Telemetry Preprocessing, Feature Engineering & Affine Normalization.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. FEATURE EXTRACTION PIPELINE:
   Raw temporal and spatial events are transformed into structured feature vectors:
     - Efficiency Ratio: eta = |path_optimal| / |path_player| in (0, 1].
       Measures topological optimality and dead-end avoidance.
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
   Ensuring isotropic weight across all behavioral dimensions.

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
    logger = RoundLogger(db_path=db_path) if db_path else RoundLogger()
    rows = logger.fetch_all()
    df = pd.DataFrame(rows)
    return df


def _repair_or_drop(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rounds with no movement at all (player quit before any
    logged decision) — these carry no signal and would produce
    div-by-zero efficiency ratios. Also drop rows with invalid outcomes."""
    before = len(df)
    df = df[df["player_path_len"] > 0].copy()
    df = df.dropna(subset=["time_taken_s", "player_path_len", "optimal_path_len"])
    df = df[df["outcome"].isin(OUTCOME_MAP.keys())].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[preprocess] dropped {dropped} incomplete round(s)")
    return df


def extract_round_features(row: dict) -> dict:
    """Extract engineered features for a single round dict (e.g. from DB)
    matching the pipeline's expected format."""
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
    optimal_len = row.get("optimal_path_len", path_len)
    outcome_str = row.get("outcome", "timeout")

    return {
        "time_taken_s": float(row.get("time_taken_s", 0.0)),
        "efficiency_ratio": float(optimal_len / path_len),
        "backtracks": int(row.get("backtracks", 0)),
        "avg_decision_time_s": float(row.get("avg_decision_time_s", 0.0)),
        "decision_time_variance": float(np.var(times)) if len(times) > 1 else 0.0,
        "outcome_code": int(OUTCOME_MAP.get(outcome_str, OUTCOME_MAP["timeout"])),
        "maze_size": int(row.get("maze_width", 0) * row.get("maze_height", 0)),
        "dead_end_count": int(row.get("dead_end_count", 0)),
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _repair_or_drop(df)

    df["efficiency_ratio"] = df["optimal_path_len"] / df["player_path_len"]
    df["maze_size"] = df["maze_width"] * df["maze_height"]
    df["outcome_code"] = df["outcome"].map(OUTCOME_MAP)

    def _decision_var(raw_json):
        try:
            times = json.loads(raw_json) if raw_json else []
        except (TypeError, json.JSONDecodeError):
            times = []
        return float(np.var(times)) if len(times) > 1 else 0.0

    df["decision_time_variance"] = df["decision_times_s"].apply(_decision_var)

    return df


def min_max_normalize(df: pd.DataFrame, columns: list) -> tuple:
    """Returns (normalised_df, scale_params) so the same scale can be
    reapplied to a single live round during play (for the win-probability
    meter and KNN skill lookup)."""
    scale_params = {}
    out = df.copy()
    for col in columns:
        lo, hi = out[col].min(), out[col].max()
        span = hi - lo if hi != lo else 1.0
        out[col] = (out[col] - lo) / span
        scale_params[col] = {"min": float(lo), "max": float(hi)}
    return out, scale_params


def apply_scale(row: dict, scale_params: dict) -> dict:
    """Apply a previously-fit scale to a single new feature dict (e.g.
    live during a round) — preserves unscaled features while scaling
    columns present in scale_params."""
    scaled = dict(row)
    for col, params in scale_params.items():
        if col in row:
            span = params["max"] - params["min"] or 1.0
            scaled[col] = (row[col] - params["min"]) / span
    return scaled


def build_training_frame(db_path: str | None = None) -> tuple:
    """Full pipeline: raw rows -> engineered -> normalised.
    Returns (normalized_df, scale_params)."""
    raw = load_raw_dataframe(db_path)
    if raw.empty:
        raise ValueError("No rounds logged yet — run data/simulate.py first")
    engineered = engineer_features(raw)
    encode_cols = ["outcome_code"]  # categorical, encoded not scaled
    numeric_cols = RAW_NUMERIC_FEATURES + ["decision_time_variance"]
    normalized, scale_params = min_max_normalize(engineered, numeric_cols)
    keep = numeric_cols + encode_cols + ["skill_label", "player_tag", "session_id"]
    return normalized[keep], scale_params


if __name__ == "__main__":
    df, scale = build_training_frame()
    print(df.describe())
    print(f"\n{len(df)} rows ready for modelling. Skill label counts:")
    print(df["skill_label"].value_counts())
