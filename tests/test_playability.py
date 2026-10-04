from dataclasses import asdict

import pygame
import pytest

from data.logger import RoundLogger, RoundRecord
from game.astar import astar_path, Hunter, HunterState
from game.engine import RoundConfig, RoundEngine, Outcome
from game.session import RoundSession, configure_round
from game.ui.controls import HeldMovement


def test_held_keys_repeat_and_release_without_os_repeat():
    movement = HeldMovement()
    assert movement.press(pygame.K_RIGHT, 1.0) == "E"
    assert movement.update(1.1) is None
    assert movement.update(1.23) == "E"
    assert movement.press(pygame.K_RIGHT, 1.24) is None
    assert movement.update(1.4) == "E"
    assert movement.press(pygame.K_DOWN, 1.41) == "S"
    assert movement.update(1.64) == "S"
    movement.release(pygame.K_DOWN)
    assert movement.update(1.8) == "E"
    movement.clear()
    assert movement.update(3.0) is None


@pytest.mark.parametrize("ability", ["walk", "dash"])
def test_smoke_allows_safe_passage_past_hunter(ability):
    engine = RoundEngine(RoundConfig(width=8, height=8, seed=42))
    engine.update(simulated_dt=0)
    destination = next(engine.maze.neighbors(*engine.player_pos))
    direction = next(d for d, delta in engine.maze.DIRS.items() if delta == destination)
    engine.hunter.position = destination
    engine.hunter.state = HunterState.PURSUIT
    assert engine.use_ability("smoke")
    if ability == "walk":
        assert engine.move_player(direction)
    else:
        assert engine.use_ability("dash", direction=direction)
    assert engine.outcome == Outcome.PLAYING
    assert not engine.hunter.caught(engine.player_pos)


def test_active_hunter_still_catches_player():
    engine = RoundEngine(RoundConfig(width=8, height=8, seed=42))
    engine.update(simulated_dt=0)
    destination = next(engine.maze.neighbors(*engine.player_pos))
    direction = next(d for d, delta in engine.maze.DIRS.items() if delta == destination)
    engine.hunter.position = destination
    engine.hunter.state = HunterState.PURSUIT
    engine.move_player(direction)
    assert engine.outcome == Outcome.CAUGHT


def test_hunter_uses_corridor_distance_for_detection():
    engine = RoundEngine(RoundConfig(width=8, height=8, seed=42))
    nearby_but_blocked = next((x,y) for y in range(8) for x in range(8)
                             if x+y <= 8 and engine.maze.shortest_path_len((0,0),(x,y)) > 8)
    hunter = Hunter((0,0), move_interval=100)
    hunter.update(engine.maze, nearby_but_blocked)
    assert hunter.state == HunterState.PATROL


def test_route_guidance_updates_with_progress():
    engine = RoundEngine(configure_round(RoundConfig(seed=42), "Recruit"))
    engine.update(simulated_dt=0)
    guide = engine.route_guide_path()
    assert len(guide) == 6
    assert engine.route_progress == 0
    start, target = engine.player_pos, guide[0]
    direction = next(d for d, delta in engine.maze.DIRS.items() if delta == (target[0]-start[0],target[1]-start[1]))
    engine.move_player(direction)
    assert engine.route_guide_path()[0] == guide[1]
    assert engine.route_progress > 0


def test_spawn_does_not_block_escape_route_when_side_corridors_exist():
    engine = RoundEngine(configure_round(RoundConfig(seed=42), "Recruit"))
    route = astar_path(engine.maze, engine.maze.start, engine.maze.exit)
    assert engine.hunter.position not in route


def test_adaptive_helps_after_two_losses(tmp_path):
    logger = RoundLogger(str(tmp_path / "rounds.db"))
    for _ in range(2):
        logger.log_round(RoundRecord(session_id="s", player_tag="alice", is_bot=False,
                         maze_width=14, maze_height=14, optimal_path_len=70, dead_end_count=10,
                         player_path_len=20, time_taken_s=20, backtracks=2,
                         outcome="caught", predicted_skill="Expert"))
    session = RoundSession(RoundConfig(seed=42), "alice", logger)
    engine = session.start("Adaptive")
    assert engine.config.route_guide
    assert engine.config.hunter_move_interval >= 60
    assert engine.config.awakening_duration_s >= 8
    assert engine.config.round_time_limit_s > 60
    custom = session.start("Custom")
    assert asdict(custom.config) == asdict(session.base_config)


def test_recruit_is_winnable_at_a_deliberate_pace():
    """No teleporting or instant movement: walk the displayed guide with real hunter ticks."""
    escapes = 0
    for seed in range(30):
        engine = RoundEngine(configure_round(RoundConfig(seed=seed), "Recruit"))
        engine.update(simulated_dt=0)
        while engine.outcome == Outcome.PLAYING:
            engine.update(simulated_dt=0.4)
            if engine.outcome != Outcome.PLAYING:
                break
            target = engine.route_guide_path()[0]
            x,y = engine.player_pos
            direction = next(d for d, delta in engine.maze.DIRS.items() if delta == (target[0]-x,target[1]-y))
            engine.move_player(direction)
        escapes += engine.outcome == Outcome.ESCAPED
    assert escapes >= 27, f"Recruit should permit an escape on most guided runs; got {escapes}/30"
