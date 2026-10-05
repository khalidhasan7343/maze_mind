"""Verify isometric projection, level selection, first-move pursuit, controls, persistence,
and Recruit balance. These checks protect the new interface while exercising the shared
engine.

Isometric input, first-move activation, pursuit, and shared round persistence.
"""

from dataclasses import replace
from unittest.mock import Mock
import sys

import pygame
import pytest

import main
from data.logger import RoundLogger
from game.astar import HunterState, astar_path
from game.engine import Outcome, RoundConfig, RoundEngine
from game.session import configure_round
from game.ui import isometric_ui as ui
from game.ui.isometric_render import IsoProjection


def toward(engine, cell):
    """Translate a target neighbor into a grid direction so tests can move legally through
    generated mazes without hard-coded routes.
    """
    x, y = engine.player_pos
    return next(d for d, delta in engine.maze.DIRS.items() if delta == (cell[0] - x, cell[1] - y))


@pytest.fixture
def game(tmp_path):
    """Create an isolated isometric controller with prediction disabled so UI lifecycle
    tests do not depend on local model artifacts.
    """
    game = ui.IsometricGame(RoundConfig(seed=42), "alice", RoundLogger(str(tmp_path / "rounds.db")))
    for predictor in (game.session.skill_predictor, game.session.style_predictor, game.session.win_predictor):
        predictor.available = False
    return game


@pytest.fixture
def interface(game):
    """Provide and clean up a Pygame display so screen and input tests can exercise the
    real interface safely under dummy SDL drivers.
    """
    pygame.init()
    view = ui.Interface(pygame.display.set_mode((1280, 840)), game)
    yield view
    pygame.quit()


@pytest.mark.parametrize("dimensions", [(8, 8), (14, 14), (9, 7)])
def test_floor_clicks_invert_projection_for_every_cell(dimensions):
    """Check every floor-cell center across grid shapes so mouse clicks select the correct
    logical cell in the isometric view.
    """
    width, height = dimensions
    projection = IsoProjection.fit(pygame.Rect(48, 190, 876, 562), width, height)
    for y in range(height):
        for x in range(width):
            center = projection.project(x + .5, y + .5)
            assert projection.cell_at(center, width, height) == (x, y)
    assert projection.cell_at(projection.project(-1, -1), width, height) is None


def test_menu_has_no_active_round_and_seed_zero_is_preserved(tmp_path):
    """Protect delayed round creation and seed zero so browsing the menu cannot start the
    timer or change a requested reproducible layout.
    """
    game = ui.IsometricGame(RoundConfig(seed=0), "alice", RoundLogger(str(tmp_path / "zero.db")))
    assert game.state == "menu" and game.engine is None
    maze, config = game.preview()
    expected = RoundEngine(configure_round(config, "Recruit")).maze
    assert maze.walls == expected.walls


@pytest.mark.parametrize("tier", ["Recruit", "Runner", "Master", "Adaptive", "Custom"])
def test_every_level_waits_until_a_legal_move(game, tier):
    """Check first-move activation across all levels so idle time and blocked movement
    cannot begin the chase.
    """
    game.selected_tier = tier
    game.start()
    e = game.engine
    initial_hunter = e.hunter.position
    e.update(simulated_dt=1000)
    assert e.elapsed_s == 0 and e.frame == 0
    assert e.hunter.position == initial_hunter
    assert not e.move_player("N")  # The top boundary is always closed.
    assert not e.chase_started
    neighbor = next(e.maze.neighbors(*e.player_pos))
    assert e.move_player(toward(e, neighbor))
    assert e.chase_started and e.hunter.state == HunterState.PURSUIT
    assert e.config.awakening_duration_s == 0
    e.update(simulated_dt=1 / e.config.fps)
    assert e.frame == 1 and e.elapsed_s > 0


def test_dash_can_start_the_chase_but_radar_does_not(game):
    """Protect movement-based activation so a dash begins play while a stationary radar use
    leaves the round waiting.
    """
    game.start()
    e = game.engine
    e.update(simulated_dt=0)
    assert e.use_ability("radar")
    assert not e.chase_started
    assert e.use_ability("dash")
    assert e.chase_started


def test_waiting_wall_time_is_excluded_from_timer_and_first_decision(monkeypatch):
    """Check the first-move clock reset so time spent waiting is absent from round duration
    and decision telemetry.
    """
    clock = [0.0]
    monkeypatch.setattr("game.engine.time.monotonic", lambda: clock[0])
    e = RoundEngine(RoundConfig(width=8, height=8, seed=42, hunter_waits_for_player=True))
    clock[0] = 500
    assert e.move_player(toward(e, next(e.maze.neighbors(*e.player_pos))))
    assert e.elapsed_s == 0 and e.decision_times_s == [0.0]
    clock[0] = 501
    e.update()
    assert e.elapsed_s == pytest.approx(1)


def test_hunter_pursues_beyond_detection_radius_and_moves_toward_player(game):
    """Protect continuous isometric pursuit so the hunter follows the player even beyond
    the classic patrol detection range.
    """
    game.start()
    e = game.engine
    e.update(simulated_dt=0)
    e.move_player(toward(e, next(e.maze.neighbors(*e.player_pos))))
    before = e.maze.shortest_path_len(e.hunter.position, e.player_pos)
    assert before > e.hunter.detection_radius
    e.update(simulated_dt=e.config.hunter_move_interval / e.config.fps)
    assert e.hunter.state == HunterState.PURSUIT
    assert e.maze.shortest_path_len(e.hunter.position, e.player_pos) == before - 1
    assert e.hunter._target == e.player_pos


def test_smoke_still_freezes_active_isometric_hunter(game):
    """Protect stun behavior in the new view so immediate pursuit still allows the player
    to escape using smoke.
    """
    game.start()
    e = game.engine
    e.update(simulated_dt=0)
    e.move_player(toward(e, next(e.maze.neighbors(*e.player_pos))))
    before = e.hunter.position
    assert e.use_ability("smoke")
    e.update(simulated_dt=2.9)
    assert e.hunter.position == before and e.hunter.is_stunned
    e.update(simulated_dt=.2)
    assert e.hunter.state == HunterState.PURSUIT


def test_level_selection_click_move_pause_focus_and_resume(interface, game):
    """Exercise real input transitions so selecting a level, tile movement, and focus-
    triggered pause keep UI and engine state synchronized.
    """
    view = interface
    view.draw()
    assert set(k for k in view.buttons if k.startswith("level:")) == {f"level:{tier}" for tier in ui.TIERS}
    view.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                  pos=view.buttons["level:Runner"].center), 0)
    assert game.selected_tier == "Runner" and game.engine is None
    view.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN), 0)
    view.draw()
    e = game.engine
    e.update(simulated_dt=0)
    neighbor = next(e.maze.neighbors(*e.player_pos))
    point = view.projection.project(neighbor[0] + .5, neighbor[1] + .5)
    view.event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=point), 0)
    assert e.player_pos == neighbor and e.chase_started
    game.held.press(pygame.K_d, 0)
    view.event(pygame.event.Event(pygame.WINDOWFOCUSLOST), 0)
    assert game.state == "paused" and not game.held.keys
    e.update(simulated_dt=1000)
    assert e.elapsed_s == 0
    view.draw()
    assert set(view.buttons) == {"pause", "menu"}
    view.event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN), 0)
    assert game.state == "playing" and not e.is_paused


@pytest.mark.parametrize("outcome", [Outcome.ESCAPED, Outcome.CAUGHT, Outcome.TIMEOUT])
def test_result_screens_save_once_and_allow_level_selection(interface, game, outcome):
    """Protect every terminal result, one-time persistence, and replay reset so completed
    rounds do not create duplicate history entries.
    """
    game.start()
    e = game.engine
    e.update(simulated_dt=0)
    if outcome == Outcome.TIMEOUT:
        e.move_player(toward(e, next(e.maze.neighbors(*e.player_pos))))
        e.use_ability("smoke")
        e.config = replace(e.config, round_time_limit_s=.1)
        e.update(simulated_dt=.2)
    else:
        if outcome == Outcome.ESCAPED:
            path = astar_path(e.maze, e.player_pos, e.maze.exit)
            e.player_pos = path[-2]
            e.hunter.position = e.maze.start
            target = e.maze.exit
        else:
            target = next(e.maze.neighbors(*e.player_pos))
            e.hunter.position = target
        e.move_player(toward(e, target))
    for _ in range(3):
        game.update(0)
        interface.draw()
    assert e.outcome == outcome and game.state == "results"
    assert game.session.logger.count() == 1
    assert game.session.logger.fetch_all()[0]["outcome"] == outcome.value
    interface.action("menu")
    interface.draw()
    assert game.state == "menu"
    game.start()
    assert not game.engine.chase_started
    assert game.session.logger.count() == 1


@pytest.mark.parametrize("view_args", [[], ["--view", "isometric"]])
def test_cli_can_launch_isometric_view(tmp_path, monkeypatch, view_args):
    """Check home-selected and explicit isometric routing so both launch paths forward
    the selected difficulty and player tag.
    """
    logger = RoundLogger(str(tmp_path / "cli.db"))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    play = Mock(return_value=Outcome.PLAYING)
    monkeypatch.setattr(ui, "run_round", play)
    from game.ui import home
    chooser = Mock(return_value="isometric")
    monkeypatch.setattr(home, "choose_view", chooser)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", *view_args, "--difficulty", "Runner", "--tag", "alice"])
    main.main()
    assert play.call_args.kwargs["initial_tier"] == "Runner"
    assert play.call_args.kwargs["player_tag"] == "alice"
    assert chooser.call_count == (0 if view_args else 1)


def test_isometric_event_loop_starts_with_menu_then_moves(tmp_path, monkeypatch):
    """Run the real event loop with scheduled inputs so menu-first startup and first-move
    activation are tested together.
    """
    logger = RoundLogger(str(tmp_path / "ui.db"))
    reference = RoundEngine(configure_round(RoundConfig(seed=42), "Recruit"))
    direction = toward(reference, next(reference.maze.neighbors(*reference.player_pos)))
    key = next(k for k, d in ui.KEY_TO_DIR.items() if d == direction)
    schedule = [[], [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
                [pygame.event.Event(pygame.KEYDOWN, key=key)], [pygame.event.Event(pygame.QUIT)]]
    monkeypatch.setattr(pygame.event, "get", lambda: schedule.pop(0))
    # Capture actual menu and play draws without replacing their rendering.
    observed = []
    original_draw = ui.Interface.draw
    def draw(view):
        """Observe screen state while preserving actual rendering so the integration test
        can confirm that no engine exists during the initial menu.
        """
        observed.append((view.game.state, view.game.engine))
        original_draw(view)
    monkeypatch.setattr(ui.Interface, "draw", draw)
    result = ui.run_round(RoundConfig(seed=42), logger=logger)
    assert result == Outcome.PLAYING
    assert observed[0] == ("menu", None)
    engine = next(engine for state, engine in observed if state == "playing")
    assert engine.chase_started and len(engine.path_taken) == 2
    assert logger.count() == 0


def test_isometric_recruit_remains_winnable(game):
    """Play thirty seeded guided rounds at a deliberate pace so balance changes cannot make
    Recruit routinely impossible to escape.
    """
    escapes = 0
    for seed in range(30):
        game.session.base_config = replace(game.session.base_config, seed=seed)
        game.start()
        e = game.engine
        e.update(simulated_dt=0)
        while e.outcome == Outcome.PLAYING:
            e.move_player(toward(e, e.route_guide_path()[0]))
            e.update(simulated_dt=.4)
        escapes += e.outcome == Outcome.ESCAPED
    assert escapes >= 27, f"Only {escapes}/30 guided Recruit runs escaped"
