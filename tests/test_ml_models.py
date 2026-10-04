import os
import tempfile
import pytest
import numpy as np

from data.logger import RoundLogger, RoundRecord
from data.preprocess import extract_round_features, apply_scale
from game.engine import RoundEngine, RoundConfig, Outcome
from game.astar import Hunter, astar_path
from game.maze_gen import generate_maze


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_extract_round_features():
    raw_row = {
        "time_taken_s": 15.5,
        "player_path_len": 20,
        "optimal_path_len": 16,
        "backtracks": 3,
        "avg_decision_time_s": 0.45,
        "decision_times_s": "[0.4, 0.5, 0.45]",
        "outcome": "escaped",
        "maze_width": 12,
        "maze_height": 12,
        "dead_end_count": 5,
    }
    feats = extract_round_features(raw_row)
    assert feats["time_taken_s"] == 15.5
    assert feats["efficiency_ratio"] == pytest.approx(16 / 20)
    assert feats["backtracks"] == 3
    assert feats["avg_decision_time_s"] == 0.45
    assert feats["outcome_code"] == 0
    assert feats["maze_size"] == 144
    assert feats["dead_end_count"] == 5
    assert feats["decision_time_variance"] >= 0.0


def test_apply_scale_preserves_unscaled_features():
    scale_params = {
        "time_taken_s": {"min": 0.0, "max": 60.0},
        "efficiency_ratio": {"min": 0.0, "max": 1.0},
    }
    raw = {
        "time_taken_s": 30.0,
        "efficiency_ratio": 0.8,
        "outcome_code": 1,
        "backtracks": 4,
    }
    scaled = apply_scale(raw, scale_params)
    assert scaled["time_taken_s"] == pytest.approx(0.5)
    assert scaled["efficiency_ratio"] == pytest.approx(0.8)
    assert scaled["outcome_code"] == 1
    assert scaled["backtracks"] == 4


def test_engine_simulated_dt():
    cfg = RoundConfig(width=6, height=6, round_time_limit_s=2.0, fps=10)
    engine = RoundEngine(cfg)
    assert engine.elapsed_s == 0.0

    dt = 0.5
    engine.update(simulated_dt=dt)
    assert engine.elapsed_s == pytest.approx(0.5)
    assert engine.outcome == Outcome.PLAYING

    # Step past time limit
    for _ in range(4):
        engine.update(simulated_dt=dt)
    assert engine.outcome == Outcome.TIMEOUT


def test_hunter_replans_when_path_depleted():
    maze = generate_maze(6, 6, seed=123)
    hunter = Hunter(position=(0, 0), replan_interval=100, move_interval=1)
    hunter.detection_radius = 100
    initial_path = astar_path(maze, (0, 0), (3, 3))[:2]
    hunter._path = initial_path.copy()

    # Frame 1: moves to (1, 0), path depleted to [(1, 0)] (len 1)
    hunter.update(maze, player_position=(3, 3))
    assert hunter.position == initial_path[1]
    assert len(hunter._path) == 1

    # Frame 2: since len(_path) <= 1, it should replan immediately even though frame != replan_interval
    hunter.update(maze, player_position=(3, 3))
    assert len(hunter._path) > 1


def test_logger_update_skill_label_and_fetch_by_player(temp_db):
    logger = RoundLogger(db_path=temp_db)
    rec1 = RoundRecord(
        session_id="s1", player_tag="bob", is_bot=False,
        maze_width=8, maze_height=8, optimal_path_len=10,
        dead_end_count=2, player_path_len=12, time_taken_s=8.0,
        backtracks=1, decision_times_s=[0.3, 0.4], outcome="escaped",
        skill_label="Beginner",
    )
    rid = logger.log_round(rec1)

    rec2 = RoundRecord(
        session_id="s2", player_tag="alice", is_bot=False,
        maze_width=8, maze_height=8, optimal_path_len=10,
        dead_end_count=2, player_path_len=12, time_taken_s=8.0,
        backtracks=1, decision_times_s=[0.3, 0.4], outcome="escaped",
        skill_label="Beginner",
    )
    logger.log_round(rec2)

    # Test fetch_by_player
    bobs = logger.fetch_by_player("bob")
    assert len(bobs) == 1
    assert bobs[0]["player_tag"] == "bob"

    # Test update_skill_label
    logger.update_skill_label(rid, "Expert")
    bobs_updated = logger.fetch_by_player("bob")
    assert bobs_updated[0]["skill_label"] == "Expert"


def test_predict_skill_and_style_live(trained_models):
    from ml.knn_skill import predict_skill
    from ml.kmeans_strategy import predict_style
    from ml.logreg_winprob import predict_win_probability

    features = {"time_taken_s": 15.0, "efficiency_ratio": 0.85, "backtracks": 1,
                "avg_decision_time_s": 0.25, "decision_time_variance": 0.02, "outcome_code": 0}
    assert predict_skill(features, trained_models["knn"]) in ("Beginner", "Intermediate", "Expert")
    assert isinstance(predict_style(features, trained_models["kmeans"]), str)
    probability = predict_win_probability({"dist_to_hunter": 6, "dist_to_exit": 3,
                                           "time_left_ratio": 0.75, "efficiency": 0.9}, trained_models["logreg"])
    assert 0 <= probability <= 1
