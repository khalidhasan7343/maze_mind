"""Verify connected and reproducible mazes, shortest routes, and hunter state timers. These
checks protect the algorithms that make the maze solvable and the chase functional.
"""

import pytest

from game.maze_gen import generate_maze
from game.astar import astar_path, manhattan, Hunter


def test_maze_fully_connected():
    """Check reachability of every cell so generated mazes cannot trap the player in a
    disconnected region.
    """
    maze = generate_maze(10, 10, seed=1)
    # Every cell must be reachable from start (CSP connectivity constraint)
    for y in range(maze.height):
        for x in range(maze.width):
            assert maze.shortest_path_len(maze.start, (x, y)) >= 0


def test_maze_solvable_and_nontrivial():
    """Protect the minimum escape-route constraint so generated challenges have a reachable
    exit and the requested route length.
    """
    maze = generate_maze(12, 12, seed=2, min_path_ratio=1.5)
    path_len = maze.shortest_path_len(maze.start, maze.exit)
    assert path_len > 0
    theoretical_max = (maze.width - 1) + (maze.height - 1)
    assert path_len >= 1.5 * theoretical_max


@pytest.mark.parametrize("size", [(2, 2), (2, 3), (1, 1), (1, 5)])
def test_tiny_mazes_are_supported(size):
    """Check small and narrow grids so endpoint parity and route constraints do not reject
    supported edge cases.
    """
    maze = generate_maze(*size, seed=2)
    assert maze.shortest_path_len(maze.start, maze.exit) >= 0


@pytest.mark.parametrize("kwargs", [{"width": 0, "height": 3}, {"width": 3, "height": 3, "max_attempts": 0}])
def test_invalid_maze_configuration(kwargs):
    """Protect early validation so unsupported generator inputs fail with clear errors."""
    with pytest.raises(ValueError):
        generate_maze(**kwargs)


def test_maze_deterministic_with_seed():
    """Check seeded generation so bug reports, simulations, and tests can reproduce the
    same maze.
    """
    m1 = generate_maze(8, 8, seed=42)
    m2 = generate_maze(8, 8, seed=42)
    assert m1.walls == m2.walls


@pytest.mark.parametrize("size", [(5, 5), (8, 12), (15, 15)])
def test_astar_matches_bfs_shortest_path(size):
    """Compare A* against an independent BFS distance so the hunter and route hints keep
    returning shortest legal paths.
    """
    w, h = size
    maze = generate_maze(w, h, seed=7)
    path = astar_path(maze, maze.start, maze.exit)
    assert len(path) - 1 == maze.shortest_path_len(maze.start, maze.exit)


def test_astar_unreachable_same_cell():
    """Check the already-at-goal case so pathfinding returns a valid one-cell route without
    an unnecessary move.
    """
    maze = generate_maze(6, 6, seed=3)
    path = astar_path(maze, maze.start, maze.start)
    assert path == [maze.start]


def test_manhattan_distance():
    """Protect the grid heuristic values so A* uses the intended four-direction distance
    estimate.
    """
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (2, 2)) == 0


def test_hunter_replans_and_moves():
    """Check that repeated AI ticks produce movement so planning does not leave the hunter
    permanently stationary.
    """
    maze = generate_maze(10, 10, seed=5)
    hunter = Hunter(position=maze.exit, replan_interval=1, move_interval=1)
    moved_at_least_once = False
    for _ in range(50):
        if hunter.update(maze, maze.start):
            moved_at_least_once = True
        if hunter.position == maze.start:
            break
    assert moved_at_least_once


def test_hunter_difficulty_knobs_change_cadence():
    """Protect cadence updates so difficulty adjustments reach the hunter movement and
    planning settings.
    """
    hunter = Hunter(position=(0, 0), replan_interval=5, move_interval=10)
    hunter.set_difficulty(replan_interval=2, move_interval=3)
    assert hunter.replan_interval == 2
    assert hunter.move_interval == 3


def test_hunter_fsm_awakening_and_stun():
    """Check grace and smoke timers so the hunter remains immobile until each protective
    state expires.
    """
    from game.astar import HunterState
    maze = generate_maze(6, 6, seed=42)
    # 1. Test Awakening State
    hunter = Hunter(position=(0, 0), awakening_duration_s=1.0, fps=10)
    assert hunter.state == HunterState.AWAKENING
    assert hunter.is_awakening is True
    assert hunter.awakening_time_left_s == 1.0

    for _ in range(10):
        moved = hunter.update(maze, player_position=(5, 5))
        assert not moved  # Stays still during awakening

    assert hunter.is_awakening is False
    assert hunter.state in (HunterState.PATROL, HunterState.PURSUIT)

    # 2. Test Stun State
    hunter.stun(duration_s=1.0)
    assert hunter.is_stunned is True
    assert hunter.stun_time_left_s == 1.0

    for _ in range(10):
        moved = hunter.update(maze, player_position=(5, 5))
        assert not moved  # Immobilized during stun

    assert hunter.is_stunned is False
