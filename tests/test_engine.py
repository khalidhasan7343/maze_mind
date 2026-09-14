from game.engine import RoundEngine, RoundConfig, Outcome


def make_engine(**overrides):
    cfg_kwargs = dict(width=8, height=8, round_time_limit_s=30, seed=11)
    cfg_kwargs.update(overrides)
    return RoundEngine(RoundConfig(**cfg_kwargs))


def test_round_starts_in_playing_state():
    engine = make_engine()
    assert engine.outcome == Outcome.PLAYING
    assert engine.player_pos == engine.maze.start


def test_illegal_move_into_wall_is_rejected():
    engine = make_engine()
    x, y = engine.player_pos
    # find a direction that is walled off
    walled_dirs = [d for d, is_wall in engine.maze.walls[y][x].items() if is_wall]
    assert walled_dirs, "expected at least one wall at the start cell"
    before = engine.player_pos
    moved = engine.move_player(walled_dirs[0])
    assert moved is False
    assert engine.player_pos == before


def test_reaching_exit_sets_escaped():
    engine = make_engine()
    # Walk the true shortest path to the exit and confirm ESCAPED triggers.
    from game.astar import astar_path
    path = astar_path(engine.maze, engine.maze.start, engine.maze.exit)
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        for d, (ddx, ddy) in engine.maze.DIRS.items():
            if (ddx, ddy) == (dx, dy):
                engine.move_player(d)
                break
        if engine.outcome != Outcome.PLAYING:
            break
    assert engine.outcome == Outcome.ESCAPED


def test_timeout_triggers_after_time_limit():
    engine = make_engine(round_time_limit_s=0.0)
    engine.update()
    assert engine.outcome == Outcome.TIMEOUT


def test_backtrack_is_counted():
    engine = make_engine()
    x, y = engine.player_pos
    open_dirs = [d for d, is_wall in engine.maze.walls[y][x].items() if not is_wall]
    assert open_dirs
    d = open_dirs[0]
    opposite = engine.maze.OPPOSITE[d]
    engine.move_player(d)
    if engine.outcome == Outcome.PLAYING:
        engine.move_player(opposite)
        assert engine.backtracks == 1


def test_win_probability_features_shape():
    engine = make_engine()
    feats = engine.win_probability_features()
    for key in ("dist_to_hunter", "dist_to_exit", "time_left_ratio", "efficiency"):
        assert key in feats
    assert 0.0 <= feats["time_left_ratio"] <= 1.0


def test_player_tactical_abilities():
    from game.astar import HunterState
    engine = make_engine()

    # Smoke Ability
    assert engine.player.can_smoke
    assert engine.use_ability("smoke") is True
    assert engine.hunter.state == HunterState.STUNNED
    assert engine.player.smoke_charges == 0
    assert engine.use_ability("smoke") is False  # Charges exhausted

    # Radar Ability
    assert engine.player.can_radar
    radar_info = engine.use_ability("radar")
    assert isinstance(radar_info, dict)
    assert "distance_to_exit" in radar_info
    assert "exit_heading" in radar_info
    assert "guide_path" in radar_info
    assert isinstance(radar_info["guide_path"], list)
    assert engine.player.radar_charges == 1

    # Dash Ability
    open_dirs = [d for d, is_wall in engine.maze.walls[0][0].items() if not is_wall]
    if open_dirs:
        dash_dir = open_dirs[0]
        assert engine.player.can_dash
        res = engine.use_ability("dash", direction=dash_dir)
        if res:
            assert not engine.player.can_dash
            assert engine.player.dash_cooldown_remaining_s > 0.0


def test_engine_pause_and_resume():
    engine = make_engine()
    assert not engine.is_paused

    # Toggle pause
    assert engine.toggle_pause() is True
    assert engine.is_paused is True

    # When paused, updates should not advance simulation
    initial_frame = engine.frame
    engine.update(simulated_dt=1.0)
    assert engine.frame == initial_frame

    # Resume
    assert engine.toggle_pause() is False
    assert engine.is_paused is False

    engine.update(simulated_dt=1.0)
    assert engine.frame == initial_frame + 1

