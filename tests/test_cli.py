"""Verify command argument forwarding and manual-label behavior. These checks protect the
public CLI that launches the game and updates reviewed training labels.
"""

import sys
from unittest.mock import Mock

import pytest

import main
from data.logger import RoundLogger, RoundRecord
from game.engine import Outcome


def test_play_cli_preserves_custom_parameters(tmp_path, monkeypatch):
    """Protect CLI argument forwarding so the classic launcher receives the requested
    custom grid, timer, seed, and player tag.
    """
    from game.ui import pygame_ui
    logger = RoundLogger(str(tmp_path / "cli.db"))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    play = Mock(return_value=Outcome.PLAYING)
    monkeypatch.setattr(pygame_ui, "run_round", play)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", "--view", "classic", "--tag", "alice", "--difficulty", "Custom",
                                      "--width", "9", "--height", "7", "--time-limit", "17", "--seed", "42"])
    main.main()
    config = play.call_args.args[0]
    assert (config.width, config.height, config.round_time_limit_s, config.seed) == (9, 7, 17, 42)
    assert play.call_args.kwargs["initial_tier"] == "Custom"
    assert play.call_args.kwargs["player_tag"] == "alice"


def test_label_cli_sets_verified_ground_truth(tmp_path, monkeypatch):
    """Protect verified-label updates and missing-round errors so manual review remains a
    reliable source of supervised training labels.
    """
    logger = RoundLogger(str(tmp_path / "cli.db"))
    rid = logger.log_round(RoundRecord(session_id="s", player_tag="alice", is_bot=False,
                          maze_width=8, maze_height=8, optimal_path_len=10, dead_end_count=2,
                          player_path_len=12, time_taken_s=8.0, backtracks=1))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    monkeypatch.setattr(sys, "argv", ["main.py", "label", "--round-id", str(rid), "--skill", "Intermediate"])
    main.main()
    assert logger.fetch_all()[0]["skill_label"] == "Intermediate"
    assert logger.fetch_all()[0]["label_source"] == "verified"
    monkeypatch.setattr(sys, "argv", ["main.py", "label", "--round-id", "999", "--skill", "Expert"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
