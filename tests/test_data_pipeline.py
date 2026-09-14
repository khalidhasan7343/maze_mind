import os
import tempfile

import pytest

from data.logger import RoundLogger, RoundRecord
from data.preprocess import engineer_features, min_max_normalize, apply_scale
import pandas as pd


@pytest.fixture
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # RoundLogger creates it fresh
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_logger_roundtrip(temp_db):
    logger = RoundLogger(db_path=temp_db)
    logger.log_round(RoundRecord(
        session_id="s1", player_tag="alice", is_bot=False,
        maze_width=10, maze_height=10, optimal_path_len=15,
        dead_end_count=4, player_path_len=18, time_taken_s=9.5,
        backtracks=1, decision_times_s=[0.1, 0.2], outcome="escaped",
        hunter_replan_ms=500, hunter_move_ms=666, skill_label="Intermediate",
    ))
    rows = logger.fetch_all()
    assert len(rows) == 1
    assert rows[0]["player_tag"] == "alice"
    assert rows[0]["outcome"] == "escaped"
    assert logger.count() == 1

    summary = logger.get_player_summary("alice")
    assert summary["total_rounds"] == 1
    assert summary["escapes"] == 1
    assert summary["win_rate_pct"] == 100.0
    assert summary["best_time_s"] == 9.5
    assert len(summary["recent_rounds"]) == 1



def test_engineer_features_drops_zero_length_rounds():
    df = pd.DataFrame([
        {"player_path_len": 0, "optimal_path_len": 10, "maze_width": 8, "maze_height": 8,
         "time_taken_s": 5.0, "decision_times_s": "[]", "outcome": "timeout"},
        {"player_path_len": 12, "optimal_path_len": 10, "maze_width": 8, "maze_height": 8,
         "time_taken_s": 5.0, "decision_times_s": "[0.1,0.2]", "outcome": "escaped"},
    ])
    out = engineer_features(df)
    assert len(out) == 1
    assert "efficiency_ratio" in out.columns
    assert out.iloc[0]["efficiency_ratio"] == pytest.approx(10 / 12)


def test_min_max_normalize_and_apply_scale_roundtrip():
    df = pd.DataFrame({"x": [0, 5, 10]})
    normed, scale = min_max_normalize(df, ["x"])
    assert normed["x"].min() == 0.0
    assert normed["x"].max() == 1.0
    scaled_point = apply_scale({"x": 5}, scale)
    assert scaled_point["x"] == pytest.approx(0.5)


def test_min_max_normalize_constant_column_no_division_by_zero():
    df = pd.DataFrame({"x": [3, 3, 3]})
    normed, scale = min_max_normalize(df, ["x"])
    assert not normed["x"].isna().any()


def test_cmd_records_cli(capsys):
    import argparse
    from main import cmd_records
    args = argparse.Namespace(tag="player1")
    cmd_records(args)
    captured = capsys.readouterr()
    assert "OPERATIVE CAREER RECORDS" in captured.out
    assert "Operator Tag" in captured.out
