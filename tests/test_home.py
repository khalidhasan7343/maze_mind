"""Verify presentation selection before a round exists.

These tests protect home navigation, clean window shutdown, launch routing, and
configuration forwarding so adding a view chooser does not alter round records.
"""

import sys
from unittest.mock import Mock

import pygame
import pytest

import main
from data.logger import RoundLogger
from game.engine import Outcome
from game.ui import home, isometric_ui, pygame_ui
from game.ui.navigation import NavigationAction


@pytest.fixture
def picker():
    """Provide a real headless home layout so event tests use the same clickable
    geometry the player sees rather than duplicating card coordinates.
    """
    pygame.init()
    picker = home.ViewPicker(pygame.display.set_mode((1280, 840)), "alice", seed=0)
    picker.draw()
    yield picker
    pygame.quit()


@pytest.mark.parametrize("view", home.VIEWS)
@pytest.mark.parametrize("target", ["card", "button"])
def test_home_card_and_button_open_the_selected_view(picker, view, target):
    """Check both click targets for each view so selecting a preview or its button
    opens the intended interface.
    """
    rect = picker.cards[view] if target == "card" else picker.buttons[view]
    event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=rect.center)
    assert picker.handle_event(event) == view


def test_home_keyboard_and_hover_selection_agree(picker):
    """Protect keyboard and mouse selection so Enter opens the currently highlighted
    view instead of always using the former default.
    """
    assert picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)) == "isometric"
    picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    assert picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)) == "classic"
    picker.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=picker.cards["isometric"].center))
    assert picker.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)) == "isometric"


@pytest.mark.parametrize("view,key", [("isometric", pygame.K_RETURN), ("classic", pygame.K_LEFT)])
def test_home_window_returns_selection_without_starting_an_engine(monkeypatch, view, key):
    """Run the real home loop so view selection closes its window cleanly while no
    playable engine is constructed before the next level screen.
    """
    engine = Mock(side_effect=AssertionError("Home must not construct a round"))
    monkeypatch.setattr("game.engine.RoundEngine", engine)
    schedule = [[], [pygame.event.Event(pygame.KEYDOWN, key=key)]]
    if view == "classic":
        schedule.append([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)])
    monkeypatch.setattr(pygame.event, "get", lambda: schedule.pop(0))
    assert home.choose_view("alice", seed=0) == view
    engine.assert_not_called()
    assert not pygame.get_init()


@pytest.mark.parametrize("event", [pygame.event.Event(pygame.QUIT),
                                  pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
                                  pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=(1190, 45))])
def test_home_can_close_without_a_view_choice(monkeypatch, event):
    """Check all visible cancellation paths so leaving Home cannot accidentally
    launch either view or leave Pygame initialized.
    """
    monkeypatch.setattr(pygame.event, "get", lambda: [event])
    assert home.choose_view() is None
    assert not pygame.get_init()


@pytest.mark.parametrize("view", home.VIEWS)
def test_default_play_opens_home_and_forwards_custom_settings(tmp_path, monkeypatch, view):
    """Protect home-to-game routing and the complete custom configuration so view
    choice cannot reset the player's difficulty, dimensions, timer, or seed.
    """
    chooser = Mock(return_value=view)
    monkeypatch.setattr(home, "choose_view", chooser)
    logger = RoundLogger(str(tmp_path / "home.db"))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    classic = Mock(return_value=Outcome.PLAYING)
    isometric = Mock(return_value=Outcome.PLAYING)
    monkeypatch.setattr(pygame_ui, "run_round", classic)
    monkeypatch.setattr(isometric_ui, "run_round", isometric)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", "--tag", "alice", "--difficulty", "Custom",
                                      "--width", "9", "--height", "7", "--time-limit", "17", "--seed", "0"])
    main.main()
    chooser.assert_called_once_with(player_tag="alice", seed=0)
    selected, other = (classic, isometric) if view == "classic" else (isometric, classic)
    selected.assert_called_once()
    other.assert_not_called()
    config = selected.call_args.args[0]
    assert (config.width, config.height, config.round_time_limit_s, config.seed) == (9, 7, 17, 0)
    assert selected.call_args.kwargs["initial_tier"] == "Custom"
    assert selected.call_args.kwargs["player_tag"] == "alice"
    assert selected.call_args.kwargs["logger"] is logger


@pytest.mark.parametrize("view", home.VIEWS)
def test_explicit_view_skips_home(tmp_path, monkeypatch, view):
    """Keep direct view launches available so scripts and players can intentionally
    bypass the home screen with a --view argument.
    """
    chooser = Mock(side_effect=AssertionError("Explicit view must skip Home"))
    monkeypatch.setattr(home, "choose_view", chooser)
    monkeypatch.setattr("data.logger.RoundLogger", lambda: RoundLogger(str(tmp_path / "direct.db")))
    play = Mock(return_value=Outcome.PLAYING)
    module = pygame_ui if view == "classic" else isometric_ui
    monkeypatch.setattr(module, "run_round", play)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", "--view", view])
    main.main()
    play.assert_called_once()
    chooser.assert_not_called()


def test_cancelled_home_does_not_create_a_logger_or_launch_a_game(monkeypatch, capsys):
    """Protect a clean early exit so cancelling view selection never opens a
    player database, loads skill history, or launches a game window.
    """
    monkeypatch.setattr(home, "choose_view", Mock(return_value=None))
    logger = Mock(side_effect=AssertionError("Cancelled Home must not access records"))
    monkeypatch.setattr("data.logger.RoundLogger", logger)
    classic, isometric = Mock(), Mock()
    monkeypatch.setattr(pygame_ui, "run_round", classic)
    monkeypatch.setattr(isometric_ui, "run_round", isometric)
    monkeypatch.setattr(sys, "argv", ["main.py", "play"])
    main.main()
    logger.assert_not_called()
    classic.assert_not_called()
    isometric.assert_not_called()
    assert "Game closed." in capsys.readouterr().out


@pytest.mark.parametrize("view", home.VIEWS)
def test_real_home_transitions_to_each_view_level_menu(tmp_path, monkeypatch, view):
    """Exercise actual window cleanup and reinitialization across the whole launch
    flow so both interfaces can draw their level menus after leaving Home.
    """
    logger = RoundLogger(str(tmp_path / "transition.db"))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    keys = []
    if view == "classic":
        keys.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT))
    keys.append(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN))
    schedule = [[], keys, [], [pygame.event.Event(pygame.QUIT)]]
    captions = []

    def events():
        """Observe real window titles while providing deterministic input so the
        test can distinguish Home from the chosen level-selection window.
        """
        captions.append(pygame.display.get_caption()[0])
        return schedule.pop(0)

    monkeypatch.setattr(pygame.event, "get", events)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", "--tag", "alice", "--seed", "0"])
    main.main()
    expected = "Maze Mind / Isometric" if view == "isometric" else "Maze Mind — Tactical AI Maze Runner"
    assert captions == ["Maze Mind / Home", "Maze Mind / Home", expected, expected]
    assert logger.count() == 0
    assert not pygame.get_init()


@pytest.mark.parametrize("view", home.VIEWS)
@pytest.mark.parametrize("control", ["button", "keyboard"])
def test_each_level_menu_can_return_home(tmp_path, monkeypatch, view, control):
    """Exercise the drawn Back button and Esc in both real menus so returning
    Home closes the view cleanly without saving a round that never started.
    """
    logger = RoundLogger(str(tmp_path / "back.db"))
    from game.engine import RoundConfig
    observed = {}
    if view == "classic":
        original = pygame_ui.screens.draw_menu

        def draw_menu(**kwargs):
            """Capture the actual classic Home button so the test clicks the
            control drawn by the menu instead of duplicating its coordinates.
            """
            observed["rect"] = kwargs["btn_home_menu"].rect
            original(**kwargs)

        monkeypatch.setattr(pygame_ui.screens, "draw_menu", draw_menu)
    else:
        original = isometric_ui.Interface.draw

        def draw_menu(interface):
            """Capture the isometric Home hit area after the real menu draws so
            the mouse test uses the same control as a player.
            """
            original(interface)
            observed["rect"] = interface.buttons["home"]

        monkeypatch.setattr(isometric_ui.Interface, "draw", draw_menu)
    frame = [0]

    def events():
        """Wait for a rendered menu, then use the requested navigation control
        so the first event cannot rely on uninitialized button geometry.
        """
        frame[0] += 1
        if frame[0] == 1:
            return []
        if control == "keyboard":
            return [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)]
        return [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=observed["rect"].center)]

    monkeypatch.setattr(pygame.event, "get", events)
    module = pygame_ui if view == "classic" else isometric_ui
    assert module.run_round(RoundConfig(seed=0), logger=logger) is NavigationAction.HOME
    assert "rect" in observed
    assert logger.count() == 0
    assert not pygame.get_init()


@pytest.mark.parametrize("first_view", home.VIEWS)
@pytest.mark.parametrize("direct", [False, True])
def test_return_home_can_switch_views_and_preserve_settings(tmp_path, monkeypatch, first_view, direct):
    """Protect repeated launcher navigation, including --view launches, so
    returning Home keeps the same player, logger, and custom round settings.
    """
    second_view = "classic" if first_view == "isometric" else "isometric"
    choices = [second_view] if direct else [first_view, second_view]
    chooser = Mock(side_effect=choices)
    monkeypatch.setattr(home, "choose_view", chooser)
    logger = RoundLogger(str(tmp_path / "switch.db"))
    factory = Mock(return_value=logger)
    monkeypatch.setattr("data.logger.RoundLogger", factory)
    first = Mock(return_value=NavigationAction.HOME)
    second = Mock(return_value=Outcome.PLAYING)
    first_module = pygame_ui if first_view == "classic" else isometric_ui
    second_module = pygame_ui if second_view == "classic" else isometric_ui
    monkeypatch.setattr(first_module, "run_round", first)
    monkeypatch.setattr(second_module, "run_round", second)
    view_args = ["--view", first_view] if direct else []
    monkeypatch.setattr(sys, "argv", ["main.py", "play", *view_args, "--tag", "alice", "--difficulty", "Custom",
                                      "--width", "9", "--height", "7", "--time-limit", "17", "--seed", "0"])
    main.main()
    first.assert_called_once()
    second.assert_called_once()
    factory.assert_called_once()
    assert chooser.call_count == (1 if direct else 2)
    for play in [first, second]:
        config = play.call_args.args[0]
        assert (config.width, config.height, config.round_time_limit_s, config.seed) == (9, 7, 17, 0)
        assert play.call_args.kwargs["player_tag"] == "alice"
        assert play.call_args.kwargs["initial_tier"] == "Custom"
        assert play.call_args.kwargs["logger"] is logger
    assert logger.count() == 0


def test_real_views_can_go_back_home_and_switch_repeatedly(tmp_path, monkeypatch, capsys):
    """Drive Home, both actual level menus, and cancellation in one session so
    repeated window cleanup and recreation cannot break Back navigation.
    """
    logger = RoundLogger(str(tmp_path / "roundtrip.db"))
    monkeypatch.setattr("data.logger.RoundLogger", lambda: logger)
    enter = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
    back = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)
    left = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_LEFT)
    schedule = [[], [enter], [], [back], [], [left, enter], [], [back], [], [pygame.event.Event(pygame.QUIT)]]
    captions = []

    def events():
        """Record the active window before each scripted input batch so the
        test verifies that Back really reopened Home between view choices.
        """
        captions.append(pygame.display.get_caption()[0])
        return schedule.pop(0)

    monkeypatch.setattr(pygame.event, "get", events)
    monkeypatch.setattr(sys, "argv", ["main.py", "play", "--tag", "alice", "--seed", "0"])
    main.main()
    expected = ["Maze Mind / Home", "Maze Mind / Isometric", "Maze Mind / Home",
                "Maze Mind — Tactical AI Maze Runner", "Maze Mind / Home"]
    assert captions == [caption for caption in expected for _ in range(2)]
    assert "Game closed." in capsys.readouterr().out
    assert logger.count() == 0
    assert not pygame.get_init()
