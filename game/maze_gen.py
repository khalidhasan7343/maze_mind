"""
Constraint Satisfaction Problem (CSP) Maze Generation.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. CONSTRAINT SATISFACTION PROBLEM (CSP) FORMULATION:
   A CSP is formally defined by a triplet (X, D, C):
     - Variables X: Set of undirected passage edges E = {e_{u, v}} between every
       pair of orthogonal adjacent cells u, v in the grid.
     - Domains D: Binary valuation for each edge: D(e) = {WALL, PASSAGE}.
     - Constraints C:
       * Hard Constraint 1 (Global Connectivity): Every cell in V must be
         reachable from every other cell (Graph G = (V, E_open) is connected).
       * Hard Constraint 2 (Acyclicity / Perfection): No redundant loops exist
         (spanning tree topology).
       * Hard Constraint 3 (Solvability): Start (0, 0) and Exit (W-1, H-1) have
         a guaranteed path.
       * Soft Constraint 4 (Non-triviality): Shortest path length must be >=
         min_path_ratio * start-to-exit Manhattan distance (default stretch: 1.5).

2. SPANNING TREE THEOREM & BACKTRACKING DFS:
   Randomized depth-first search over the dual grid graph G naturally synthesizes
   a random spanning tree, biased toward long DFS corridors (not uniformly sampled).
   Because a spanning tree on N vertices contains
   precisely N - 1 edges and no cycles:
     - Connectivity is guaranteed by construction.
     - A unique simple path exists between any arbitrary pair of nodes (u, v).
   Recursive/iterative carving operates as backtracking search where selecting an
   unvisited neighbor is domain assignment, and popping exhausted nodes corresponds
   to backtracking out of dead ends.

3. COMPUTATIONAL COMPLEXITY:
   - Time Complexity: O(|V| + |E|) = O(W * H), visiting each cell once.
   - Space Complexity: O(W * H) for the explicit traversal stack and visited set.
===============================================================================
"""

from __future__ import annotations

import random
import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Maze:
    width: int
    height: int
    # walls[y][x] is a dict of which of the 4 directions have a wall
    # True == wall present, False == open passage
    walls: list = field(default_factory=list)
    start: tuple = (0, 0)
    exit: tuple = (0, 0)

    DIRS = {
        "N": (0, -1),
        "S": (0, 1),
        "E": (1, 0),
        "W": (-1, 0),
    }
    OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int):
        """Open (non-wall) neighbours of a cell."""
        for d, (dx, dy) in self.DIRS.items():
            if not self.walls[y][x][d]:
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    yield nx, ny

    def shortest_path_len(self, a: tuple, b: tuple) -> int:
        """BFS shortest path length in cells (used for the min-length
        constraint and as the 'optimal path' baseline for efficiency
        ratio features)."""
        q = deque([a])
        dist = {a: 0}
        while q:
            cur = q.popleft()
            if cur == b:
                return dist[cur]
            for nb in self.neighbors(*cur):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        return -1  # unreachable — should never happen once constraints hold

    def dead_end_count(self) -> int:
        count = 0
        for y in range(self.height):
            for x in range(self.width):
                open_dirs = sum(1 for d in self.DIRS if not self.walls[y][x][d])
                if open_dirs == 1:
                    count += 1
        return count


def _carve(maze: Maze, rng: random.Random) -> None:
    """Randomised DFS backtracking carve = the CSP backtracking search.

    Each recursive call is a partial assignment of edge variables; picking
    an unvisited neighbour is choosing a value from that variable's
    (already domain-reduced) remaining options, and returning up the call
    stack when no unvisited neighbour remains *is* the backtrack step.
    """
    stack = [(0, 0)]
    visited = {(0, 0)}
    while stack:
        x, y = stack[-1]
        candidates = []
        for d, (dx, dy) in maze.DIRS.items():
            nx, ny = x + dx, y + dy
            if maze.in_bounds(nx, ny) and (nx, ny) not in visited:
                candidates.append((d, nx, ny))
        if not candidates:
            stack.pop()  # backtrack: this branch of the CSP is exhausted
            continue
        d, nx, ny = rng.choice(candidates)
        maze.walls[y][x][d] = False
        maze.walls[ny][nx][maze.OPPOSITE[d]] = False
        visited.add((nx, ny))
        stack.append((nx, ny))


def generate_maze(
    width: int,
    height: int,
    min_path_ratio: float = 1.5,
    seed: int | None = None,
    max_attempts: int = 50,
) -> Maze:
    """Generate a maze satisfying the CSP constraints.

    min_path_ratio is a stretch factor over Manhattan distance, capped at
    the maximum path length compatible with endpoint parity on tiny grids.
    Raise an error if the requested constraint is not found within max_attempts.
    """
    if width < 1 or height < 1 or max_attempts < 1:
        raise ValueError("Dimensions and max_attempts must be positive")
    if not math.isfinite(min_path_ratio) or min_path_ratio <= 0:
        raise ValueError("min_path_ratio must be finite and positive")
    rng = random.Random(seed)
    theoretical_max = (width - 1) + (height - 1)
    max_path = width * height - 1
    max_path -= (max_path - theoretical_max) % 2
    min_len = min(max_path, math.ceil(theoretical_max * min_path_ratio))

    for _ in range(max_attempts):
        maze = Maze(
            width=width,
            height=height,
            walls=[
                [
                    {"N": True, "S": True, "E": True, "W": True}
                    for _ in range(width)
                ]
                for _ in range(height)
            ],
        )
        _carve(maze, rng)

        # Constraint: full connectivity is guaranteed by the carve (it's a
        # spanning tree). Pick start/exit as far apart as possible, then
        # verify the min-path-length constraint; regenerate if too easy.
        maze.start = (0, 0)
        maze.exit = (width - 1, height - 1)
        path_len = maze.shortest_path_len(maze.start, maze.exit)
        if path_len >= min_len:
            return maze

    raise ValueError(f"Could not generate a maze with path length >= {min_len} in {max_attempts} attempts")


if __name__ == "__main__":
    m = generate_maze(10, 10, seed=42)
    print(f"start={m.start} exit={m.exit} "
          f"shortest_path={m.shortest_path_len(m.start, m.exit)} "
          f"dead_ends={m.dead_end_count()}")
