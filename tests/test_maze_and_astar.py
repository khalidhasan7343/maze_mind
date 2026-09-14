import pytest

from game.maze_gen import generate_maze
from game.astar import astar_path, manhattan, Hunter


def test_maze_fully_connected():
    maze = generate_maze(10, 10, seed=1)
    # Every cell must be reachable from start (CSP connectivity constraint)
    for y in range(maze.height):
        for x in range(maze.width):
            assert maze.shortest_path_len(maze.start, (x, y)) >= 0


def test_maze_solvable_and_nontrivial():
    maze = generate_maze(12, 12, seed=2, min_path_ratio=0.6)
    path_len = maze.shortest_path_len(maze.start, maze.exit)
    assert path_len > 0
    theoretical_max = (maze.width - 1) + (maze.height - 1)
    assert path_len >= 0.6 * theoretical_max


def test_maze_deterministic_with_seed():
    m1 = generate_maze(8, 8, seed=42)
    m2 = generate_maze(8, 8, seed=42)
    assert m1.walls == m2.walls


@pytest.mark.parametrize("size", [(5, 5), (8, 12), (15, 15)])
def test_astar_matches_bfs_shortest_path(size):
    w, h = size
    maze = generate_maze(w, h, seed=7)
    path = astar_path(maze, maze.start, maze.exit)
    assert len(path) - 1 == maze.shortest_path_len(maze.start, maze.exit)


def test_astar_unreachable_same_cell():
    maze = generate_maze(6, 6, seed=3)
    path = astar_path(maze, maze.start, maze.start)
    assert path == [maze.start]


def test_manhattan_distance():
    assert manhattan((0, 0), (3, 4)) == 7
    assert manhattan((2, 2), (2, 2)) == 0


def test_hunter_replans_and_moves():
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
    hunter = Hunter(position=(0, 0), replan_interval=5, move_interval=10)
    hunter.set_difficulty(replan_interval=2, move_interval=3)
    assert hunter.replan_interval == 2
    assert hunter.move_interval == 3


def test_hunter_fsm_awakening_and_stun():
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

