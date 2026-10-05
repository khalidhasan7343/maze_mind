"""Preserve fixes for migration, timing, label separation, evaluation leakage, prediction
caching, and one-time logging. Each test prevents a previously addressed failure from
returning.
"""

from dataclasses import replace
import sqlite3

import joblib
import numpy as np
import pandas as pd
import pytest

from data.logger import RoundLogger, RoundRecord
from data.preprocess import extract_round_features, engineer_features, build_feature_frame, scale_frame
from data.simulate import run_bot_round, PROFILES
from game.engine import RoundConfig, RoundEngine, Outcome
from game.session import configure_round, RoundSession, OptionalPredictor


def record(**changes):
    """Build a valid record with targeted overrides so regression tests can isolate one
    persistence or feature behavior at a time.
    """
    return RoundRecord(**dict(dict(session_id="s", player_tag="alice", is_bot=False,
                                  maze_width=8, maze_height=8, optimal_path_len=10, dead_end_count=2,
                                  player_path_len=12, time_taken_s=8.0, backtracks=1, outcome="escaped"), **changes))


def test_unknown_player_has_empty_history(tmp_path):
    """Protect player-specific empty history so an unknown tag cannot receive somebody
    else's statistics.
    """
    logger = RoundLogger(str(tmp_path / "rounds.db"))
    logger.log_round(record())
    assert logger.get_player_summary("nobody")["total_rounds"] == 0
    assert logger.get_player_summary("alice")["total_rounds"] == 1
    assert logger.get_player_summary()["total_rounds"] == 1


def test_predictions_do_not_change_verified_labels(tmp_path):
    """Protect label provenance so automatic inference cannot overwrite independently
    verified training targets.
    """
    logger = RoundLogger(str(tmp_path / "rounds.db"))
    rid = logger.log_round(record(skill_label="Beginner", label_source="verified"))
    logger.update_predictions(rid, "Expert", "rusher", "v2")
    row = logger.fetch_all()[0]
    assert row["skill_label"] == "Beginner"
    assert row["label_source"] == "verified"
    assert row["predicted_skill"] == "Expert"


def test_existing_database_migration_preserves_history(tmp_path):
    """Open a legacy schema and migrate it so updates retain old rounds and distinguish old
    from current telemetry.
    """
    from pathlib import Path
    schema = Path("db/schema.sql").read_text().split("CREATE TABLE IF NOT EXISTS round_snapshots")[0]
    new_columns = ("label_source", "predicted_skill", "predicted_style", "model_version", "remaining_path_len", "telemetry_version")
    legacy = "\n".join(line for line in schema.splitlines() if not any(line.strip().startswith(c) for c in new_columns))
    db = str(tmp_path / "legacy.db")
    with sqlite3.connect(db) as conn:
        conn.executescript(legacy)
        conn.execute("""INSERT INTO rounds (session_id,player_tag,is_bot,maze_width,maze_height,
                        optimal_path_len,dead_end_count,player_path_len,time_taken_s,backtracks,
                        avg_decision_time_s,outcome,hunter_replan_ms,hunter_move_ms,skill_label)
                        VALUES ('s','alice',0,8,8,10,2,12,8,1,0,'escaped',500,600,'Expert')""")
    logger = RoundLogger(db)
    old = logger.fetch_all()[0]
    assert old["skill_label"] == "Expert"
    assert old["label_source"] == "unknown"
    assert old["telemetry_version"] == 1
    logger.log_round(record())
    assert logger.fetch_all()[-1]["telemetry_version"] == 2
    assert RoundLogger(db).count() == 2


def test_paused_actions_and_decision_times(monkeypatch):
    """Protect action blocking and clock compensation so pause duration cannot distort
    movement decisions or total round time.
    """
    now = [100.0]
    monkeypatch.setattr("game.engine.time.monotonic", lambda: now[0])
    engine = RoundEngine(RoundConfig(width=8, height=8, seed=11))
    direction = next(d for d, blocked in engine.maze.walls[0][0].items() if not blocked)
    now[0] += 0.2
    assert engine.move_player(direction)
    engine.pause()
    position = engine.player_pos
    assert not engine.move_player(engine.maze.OPPOSITE[direction])
    assert not engine.use_ability("smoke")
    now[0] += 100
    engine.resume()
    now[0] += 0.3
    assert engine.move_player(engine.maze.OPPOSITE[direction])
    assert engine.player_pos != position
    assert engine.decision_times_s == pytest.approx([0.2, 0.3])
    assert engine.elapsed_s == pytest.approx(0.5)


def test_cadence_independent_of_update_size():
    """Compare coarse and fine update intervals so simulation results remain independent of
    render or caller update frequency.
    """
    config = RoundConfig(width=8, height=8, seed=11, awakening_duration_s=0)
    slow = RoundEngine(config)
    fast = RoundEngine(config)
    slow.update(simulated_dt=2.0)
    for _ in range(120):
        fast.update(simulated_dt=1/60)
    assert fast.elapsed_s == pytest.approx(slow.elapsed_s)
    assert fast.frame == slow.frame == 60
    assert fast.hunter.position == slow.hunter.position
    assert fast.hunter.state == slow.hunter.state


def test_bot_thinking_time_advances_game_clock():
    """Protect realistic bot timing so the hunter keeps moving during decision delays and
    synthetic telemetry measures the actual game clock.
    """
    import random
    result = run_bot_round(PROFILES[0], RoundConfig(width=8, height=8, seed=42), random.Random(0))
    assert sum(result.decision_times_s) <= result.time_taken_s + 0.001
    assert result.avg_decision_time_s >= PROFILES[0].think_time_range[0]
    assert result.label_source == "synthetic"


@pytest.mark.parametrize("tier", ["Custom", "Adaptive"])
def test_presets_preserve_base_config(tier):
    """Check configuration copying so selecting another tier does not permanently overwrite
    Custom or Adaptive grid and timer settings.
    """
    base = RoundConfig(width=9, height=7, round_time_limit_s=17)
    configure_round(base, "Master")
    configured = configure_round(base, tier, "Expert")
    assert (configured.width, configured.height, configured.round_time_limit_s) == (9, 7, 17)
    assert base.hunter_move_interval == 28
    assert configured is not base


def test_skill_history_resists_one_unusual_round():
    """Protect median-based adaptation so a single exceptional prediction cannot abruptly
    change player difficulty.
    """
    from ml.knn_skill import skill_from_history
    rows = [{"predicted_skill": "Beginner"} for _ in range(4)] + [{"predicted_skill": "Expert"}]
    assert skill_from_history(rows) == "Beginner"
    assert skill_from_history([{"predicted_skill": "Expert"}]*5) == "Expert"


def test_failed_round_efficiency_measures_progress():
    """Protect progress-based features so short failed runs do not receive an artificially
    high efficiency score.
    """
    row = dict(time_taken_s=1.0, player_path_len=2, optimal_path_len=20,
               remaining_path_len=19, backtracks=0, decision_times_s="[]", outcome="caught",
               maze_width=8, maze_height=8)
    assert extract_round_features(row)["efficiency_ratio"] == 0.5
    assert engineer_features(pd.DataFrame([row])).iloc[0]["efficiency_ratio"] == 0.5
    row["remaining_path_len"] = 25
    assert extract_round_features(row)["efficiency_ratio"] == 0


def test_snapshot_split_keeps_rounds_together(trained_models):
    """Check grouped evaluation so snapshots of the same round cannot appear in both
    training and test partitions.
    """
    from ml.logreg_winprob import split_snapshot_rounds
    training, testing = split_snapshot_rounds(trained_models["snapshots"])
    assert set(training["round_id"]).isdisjoint(testing["round_id"])
    assert set(training["won"]) == set(testing["won"]) == {0, 1}


@pytest.mark.parametrize("name,module_name,predict_name", [
    ("knn", "ml.knn_skill", "predict_skill"),
    ("kmeans", "ml.kmeans_strategy", "predict_style"),
    ("logreg", "ml.logreg_winprob", "predict_win_probability"),
])
def test_prediction_cache_loads_once_and_switches_path(trained_models, monkeypatch, tmp_path, name, module_name, predict_name):
    """Protect prediction caching and path changes so repeated inference avoids disk I/O
    while an explicitly different model still loads.
    """
    import importlib
    module = importlib.import_module(module_name)
    bundle = joblib.load(trained_models[name])
    features = {c: 0.5 for c in bundle["features"]}
    copy_path = str(tmp_path / "copy.joblib")
    joblib.dump(bundle, copy_path)
    original = joblib.load
    calls = []
    def load(path):
        """Record real artifact loads so the cache test can count disk access without
        replacing the loaded model.
        """
        calls.append(path)
        return original(path)
    monkeypatch.setattr(module, "_MODEL_CACHE", None)
    monkeypatch.setattr(module.joblib, "load", load)
    predict = getattr(module, predict_name)
    results = [predict(features, trained_models[name]) for _ in range(3)]
    assert len(calls) == 1
    assert results[0] == results[1] == results[2]
    predict(features, copy_path)
    assert calls == [trained_models[name], copy_path]


def test_knn_scaler_is_fit_after_holdout_split(trained_models, monkeypatch, tmp_path):
    """Inspect scaler training inputs so evaluation data cannot influence the feature
    ranges learned for a holdout prediction.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from ml import knn_skill
    frame = build_feature_frame(trained_models["db"])
    frame = frame[frame["label_source"].isin(["verified", "synthetic"])]
    expected = train_test_split(frame[knn_skill.FEATURE_COLS].values, frame["skill_label"].values,
                                test_size=0.2, random_state=42, stratify=frame["skill_label"].values)[0]
    original = MinMaxScaler.fit
    fits = []
    def fit(self, X, y=None):
        """Capture scaler fit inputs while calling the original implementation so the test
        can verify split isolation with genuine training behavior.
        """
        fits.append(np.asarray(X).copy())
        return original(self, X, y)
    monkeypatch.setattr(MinMaxScaler, "fit", fit)
    knn_skill.train(db_path=trained_models["db"], model_path=str(tmp_path / "knn.joblib"))
    np.testing.assert_array_equal(fits[0], expected)
    assert len(fits[-1]) == len(frame)


def test_predicted_labels_excluded_from_training(trained_models, monkeypatch, tmp_path):
    """Protect supervised eligibility so prediction-only rows cannot become their own
    training labels.
    """
    from ml import knn_skill
    frame = build_feature_frame(trained_models["db"])
    frame["label_source"] = "unknown"
    frame["predicted_skill"] = frame["skill_label"]
    monkeypatch.setattr(knn_skill, "build_feature_frame", lambda _: frame)
    with pytest.raises(ValueError, match="labelled rounds"):
        knn_skill.train(db_path="ignored", model_path=str(tmp_path / "unused.joblib"))


def test_dashboard_uses_saved_model_scale(trained_models, monkeypatch, tmp_path):
    """Perturb new telemetry ranges so the dashboard must use saved normalization rather
    than redefining the trained cluster coordinates.
    """
    from dashboard import report_plots
    bundle = joblib.load(trained_models["kmeans"])
    frame = build_feature_frame(trained_models["db"])
    frame["time_taken_s"] *= 10
    expected = scale_frame(frame, bundle["scale_params"])[bundle["features"]].values
    seen = []
    original = bundle["model"].predict
    def predict(X):
        """Capture the actual clustering input so the plot test can verify that saved
        training ranges were applied.
        """
        seen.append(X.copy())
        return original(X)
    monkeypatch.setattr(bundle["model"], "predict", predict)
    monkeypatch.setattr(report_plots.joblib, "load", lambda _: bundle)
    monkeypatch.setattr(report_plots, "build_feature_frame", lambda _: frame)
    monkeypatch.setattr(report_plots, "FIG_DIR", str(tmp_path))
    report_plots.plot_pca_clusters(trained_models["kmeans"], db_path="ignored")
    np.testing.assert_array_equal(seen[0], expected)
    assert (tmp_path / "pca_clusters.png").is_file()


def test_missing_or_old_model_has_explicit_status(tmp_path):
    """Protect optional-model failure handling so missing or incompatible artifacts disable
    estimates with an understandable retraining status.
    """
    from ml import logreg_winprob
    missing = OptionalPredictor(logreg_winprob.predict_win_probability, str(tmp_path / "missing.joblib"))
    assert missing({}) is None
    assert missing.reason == "Model unavailable"
    old = str(tmp_path / "old.joblib")
    joblib.dump({"model": None}, old)
    predictor = OptionalPredictor(logreg_winprob.predict_win_probability, old)
    assert predictor({}) is None
    assert not predictor.available
    assert "retrain" in predictor.reason


def test_session_logs_once_and_keeps_ground_truth_separate(tmp_path):
    """Protect shared round finalization so logging is idempotent, snapshots get eventual
    outcomes, and predicted skill stays separate from ground truth.
    """
    logger = RoundLogger(str(tmp_path / "rounds.db"))
    session = RoundSession(RoundConfig(width=8, height=8, seed=11), "alice", logger)
    engine = session.start("Custom")
    session.skill_predictor = lambda _: "Expert"
    session.style_predictor = lambda _: "rusher"
    # Real predictor objects expose version metadata.
    session.skill_predictor.version = "skill-v2"
    session.style_predictor.version = "style-v2"
    engine.outcome = Outcome.ESCAPED
    assert session.finish()
    assert not session.finish()
    assert logger.count() == 1
    row = logger.fetch_all()[0]
    assert row["skill_label"] is None
    assert row["predicted_skill"] == "Expert"
    assert row["predicted_style"] == "rusher"
    assert logger.fetch_snapshots()[0]["won"] == 1
