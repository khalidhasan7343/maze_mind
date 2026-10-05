"""Drive classic menus and terminal screens through real event handling. These integration
checks protect navigation, rendering, custom settings, and completed-round persistence
together.
"""

from unittest.mock import Mock

import pygame
import pytest

from data.logger import RoundLogger
from game.astar import astar_path, HunterState
from game.engine import RoundConfig, RoundEngine, Outcome
from game import session as session_module
from game.ui import pygame_ui as ui


@pytest.mark.parametrize("outcome", [Outcome.ESCAPED, Outcome.CAUGHT, Outcome.TIMEOUT])
def test_terminal_outcomes_render_results_and_log_once(tmp_path, monkeypatch, outcome):
    """Exercise all classic terminal outcomes through input and rendering so results appear
    and completed rounds are stored only once.
    """
    logger = RoundLogger(str(tmp_path / "ui.db"))
    events = []
    def factory(config):
        """Prepare a real engine one step from the requested outcome so the UI test reaches
        escape, capture, or timeout through its normal event flow.
        """
        engine = RoundEngine(config)
        engine.update(simulated_dt=0)
        if outcome == Outcome.TIMEOUT:
            engine.update(simulated_dt=config.round_time_limit_s)
        else:
            if outcome == Outcome.ESCAPED:
                path = astar_path(engine.maze, engine.maze.start, engine.maze.exit)
                engine.player_pos = path[-2]
                engine.hunter.position = engine.maze.start
                target = engine.maze.exit
            else:
                target = next(engine.maze.neighbors(*engine.player_pos))
                engine.hunter.position = target
                engine.hunter.state = HunterState.PURSUIT
            x, y = engine.player_pos
            direction = next(d for d, delta in engine.maze.DIRS.items() if delta == (target[0]-x, target[1]-y))
            key = next(k for k, d in ui.KEY_TO_DIR.items() if d == direction)
            events.append(pygame.event.Event(pygame.KEYDOWN, key=key))
        return engine
    monkeypatch.setattr(session_module, "RoundEngine", factory)
    monkeypatch.setattr(pygame.event, "get", lambda: [events.pop(0)] if events else [])
    renderer = Mock(wraps=ui.screens.draw_results)
    monkeypatch.setattr(ui.screens, "draw_results", renderer)
    config = RoundConfig(width=9, height=7, round_time_limit_s=17, seed=2)
    result = ui.run_round(config, logger=logger, headless_for_testing=True, initial_tier="Custom")
    assert result == outcome
    assert renderer.call_count == 3
    assert logger.count() == 1
    assert logger.fetch_all()[0]["outcome"] == outcome.value
    assert (config.width, config.height, config.round_time_limit_s) == (9, 7, 17)


def test_menu_custom_selection_pause_and_history(tmp_path, monkeypatch):
    """Drive classic selection, pause, and history screens so navigation preserves custom
    configuration and returns to the expected state.
    """
    logger = RoundLogger(str(tmp_path / "ui.db"))
    schedule = [
        [],
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(521, 385))], # Master
        [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(845, 385))], # Custom
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)],
        [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p)],
        [pygame.event.Event(pygame.QUIT)],
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: schedule.pop(0) if schedule else [pygame.event.Event(pygame.QUIT)])
    menu = Mock(wraps=ui.screens.draw_menu)
    pause = Mock(wraps=ui.screens.draw_pause)
    history = Mock(wraps=ui.screens.draw_history)
    monkeypatch.setattr(ui.screens, "draw_menu", menu)
    monkeypatch.setattr(ui.screens, "draw_pause", pause)
    monkeypatch.setattr(ui.screens, "draw_history", history)
    config = RoundConfig(width=9, height=7, round_time_limit_s=17, seed=2)
    assert ui.run_round(config, logger=logger, initial_tier="Custom") == Outcome.PLAYING
    assert set(menu.call_args.kwargs["menu_tier_buttons"]) == {"Recruit", "Runner", "Master", "Adaptive", "Custom"}
    assert menu.call_args.kwargs["selected_tier"] == "Custom"
    assert pause.call_count == 2
    assert history.call_count == 1
    played = pause.call_args.kwargs["config"]
    assert (played.width, played.height, played.round_time_limit_s) == (9, 7, 17)
