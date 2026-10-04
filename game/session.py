"""Round lifecycle, difficulty selection, telemetry, and optional ML predictions."""

import logging
import os
from dataclasses import asdict, replace
from uuid import uuid4

from data.logger import RoundLogger, RoundRecord
from data.preprocess import extract_round_features
from game.engine import Outcome, RoundConfig, RoundEngine
from ml.artifacts import FEATURE_VERSION
from ml.knn_skill import difficulty_for_skill, skill_from_history

LOG = logging.getLogger(__name__)

DIFFICULTY_PRESETS = {
    "Recruit": dict(width=8, height=8, round_time_limit_s=120.0, hunter_replan_interval=36,
                    hunter_move_interval=60, awakening_duration_s=8.0, smoke_charges=3, radar_charges=5,
                    dash_cooldown_s=3.0, route_guide=True),
    "Runner": dict(width=12, height=12, round_time_limit_s=75.0, hunter_replan_interval=22,
                   hunter_move_interval=28, awakening_duration_s=3.0, smoke_charges=1, radar_charges=2, dash_cooldown_s=7.0),
    "Master": dict(width=14, height=14, round_time_limit_s=60.0, hunter_replan_interval=18,
                   hunter_move_interval=20, awakening_duration_s=2.0, smoke_charges=1, radar_charges=1, dash_cooldown_s=8.0),
}
TIERS = [*DIFFICULTY_PRESETS, "Adaptive", "Custom"]


def configure_round(base: RoundConfig, tier: str, skill: str = "Beginner") -> RoundConfig:
    if tier not in TIERS:
        raise ValueError(f"Unknown difficulty: {tier}")
    overrides = DIFFICULTY_PRESETS.get(tier, {})
    if tier == "Adaptive":
        overrides = difficulty_for_skill(skill)
    return replace(base, **overrides)


def tier_description(base: RoundConfig, tier: str, skill: str) -> str:
    c = configure_round(base, tier, skill)
    guide = " • Route guide ON" if c.route_guide else ""
    return f"{c.width}x{c.height} Grid • {c.round_time_limit_s:g}s Timer • {c.smoke_charges} Smokes & {c.radar_charges} Radars{guide}"


class OptionalPredictor:
    """Disable unavailable/incompatible models once, with an explicit HUD status."""

    def __init__(self, predict, model_path):
        self.predict = predict
        self.model_path = model_path
        self.available = os.path.isfile(model_path)
        self.reason = None if self.available else "Model unavailable"
        self.version = None

    def __call__(self, features):
        if not self.available:
            return None
        try:
            result = self.predict(features, model_path=self.model_path)
            # Prediction loaded the validated cache; expose its artifact version.
            import sys
            self.version = (getattr(sys.modules[self.predict.__module__], "_MODEL_CACHE", None) or {}).get("model_version")
            return result
        except (OSError, ValueError, KeyError) as exc:
            self.available = False
            self.reason = "Model unavailable — retrain"
            LOG.warning("Prediction disabled for %s: %s", self.model_path, exc)
            return None


class RoundSession:
    def __init__(self, config: RoundConfig, player_tag: str, logger: RoundLogger, skill_hint=None):
        from ml.knn_skill import predict_skill, MODEL_PATH as KNN_PATH
        from ml.kmeans_strategy import predict_style, MODEL_PATH as STYLE_PATH
        from ml.logreg_winprob import predict_win_probability, MODEL_PATH as WIN_PATH
        self.base_config = replace(config)
        self.player_tag = player_tag
        self.logger = logger
        self.skill = skill_from_history(logger.fetch_by_player(player_tag), default=skill_hint or "Beginner")
        self.skill_predictor = OptionalPredictor(predict_skill, KNN_PATH)
        self.style_predictor = OptionalPredictor(predict_style, STYLE_PATH)
        self.win_predictor = OptionalPredictor(predict_win_probability, WIN_PATH)
        self.engine = None
        self.style = None
        self.round_id = None
        self.snapshots = []
        self.next_snapshot = 0.0
        self.summary = logger.get_player_summary(player_tag)

    def start(self, tier: str) -> RoundEngine:
        history = self.logger.fetch_by_player(self.player_tag)
        self.skill = skill_from_history(history, default=self.skill)
        config = configure_round(self.base_config, tier, self.skill)
        # Losses trigger practical assistance even if the classifier predicts a higher tier.
        assistance = tier == "Adaptive" and len(history) >= 2 and all(r["outcome"] != "escaped" for r in history[-2:])
        if assistance:
            config = replace(config, hunter_move_interval=max(config.hunter_move_interval, 60),
                             awakening_duration_s=max(config.awakening_duration_s, 8.0),
                             smoke_charges=max(config.smoke_charges, 3), route_guide=True)
        self.engine = RoundEngine(config)
        if assistance:
            path_len = self.engine.maze.shortest_path_len(self.engine.maze.start, self.engine.maze.exit)
            self.engine.config = replace(config, round_time_limit_s=max(config.round_time_limit_s, path_len * 1.5 + 30))
        self.style = None
        self.round_id = None
        self.snapshots = []
        self.next_snapshot = 0.0
        self.capture_snapshot()
        return self.engine

    def capture_snapshot(self):
        if self.engine.outcome == Outcome.PLAYING and self.engine.elapsed_s >= self.next_snapshot:
            self.snapshots.append(self.engine.win_probability_features())
            self.next_snapshot = self.engine.elapsed_s + max(1.0 / self.engine.config.fps, self.engine.config.round_time_limit_s / 6)

    def finish(self) -> bool:
        """Log a terminal round once, then update predictions independently of labels."""
        e = self.engine
        if e.outcome == Outcome.PLAYING or self.round_id is not None:
            return False
        c = e.config
        record = RoundRecord(
            session_id=str(uuid4()), player_tag=self.player_tag, is_bot=False,
            maze_width=c.width, maze_height=c.height,
            optimal_path_len=e.maze.shortest_path_len(e.maze.start, e.maze.exit),
            remaining_path_len=e.maze.shortest_path_len(e.player_pos, e.maze.exit),
            dead_end_count=e.maze.dead_end_count(), player_path_len=len(e.path_taken)-1,
            time_taken_s=e.elapsed_s, backtracks=e.backtracks, decision_times_s=e.decision_times_s,
            outcome=e.outcome.value, hunter_replan_ms=round(1000*c.hunter_replan_interval/c.fps),
            hunter_move_ms=round(1000*c.hunter_move_interval/c.fps),
        )
        self.round_id = self.logger.log_round(record, self.snapshots)
        features = extract_round_features(asdict(record))
        predicted = self.skill_predictor(features)
        self.style = self.style_predictor(features)
        version = f"features-{FEATURE_VERSION}; skill={self.skill_predictor.version}; style={self.style_predictor.version}"
        self.logger.update_predictions(self.round_id, predicted, self.style, version)
        self.skill = skill_from_history(self.logger.fetch_by_player(self.player_tag), default=self.skill)
        self.summary = self.logger.get_player_summary(self.player_tag)
        return True
