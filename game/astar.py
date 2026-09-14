"""
A* Search Engine & Hunter Behavioral State Machine.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. INFORMED SEARCH:
   A* evaluates candidate paths using the evaluation function:
       f(n) = g(n) + h(n)
   where:
     - g(n) is the exact path cost from the start node to node n.
     - h(n) is the heuristic estimate of the cost from node n to the goal.

2. ADMISSIBILITY & CONSISTENCY (MONOTONICITY):
   - Admissibility: A heuristic h(n) is admissible if it never overestimates
     the true cost to the goal:
         0 <= h(n) <= h*(n)  for all n in V
   - Consistency (Triangle Inequality): For any transition from node u to v
     with edge cost c(u, v) = 1:
         h(u) <= c(u, v) + h(v)
   - Grid Movement: On a 4-directional grid with uniform step cost c = 1,
     Manhattan distance |x1 - x2| + |y1 - y2| is both strictly admissible and
     consistent. Consequently, A* is mathematically guaranteed to return an
     optimal shortest path without ever needing to re-open closed nodes.

3. COMPLEXITY ANALYSIS:
   - Time Complexity: O((|V| + |E|) log |V|) using a binary min-heap priority queue.
   - Space Complexity: O(|V|) to maintain open and closed sets.

4. BEHAVIORAL FINITE STATE MACHINE (FSM):
   Rather than acting as an unfair omniscient vacuum, the Hunter is governed by
   a 4-state behavioral machine:
     - AWAKENING: Grace period at round start granting the player time to assess
       the maze geometry and enter initial corridors.
     - PATROL: Measured, slower pacing when the player is beyond sensory range.
     - PURSUIT: High-priority dynamic A* tracking when within proximity.
     - STUNNED: Disoriented state induced by player tactical items (smoke trap).
===============================================================================
"""

from __future__ import annotations

import heapq
from enum import Enum


class HunterState(Enum):
    """Behavioral states for the Hunter AI finite state machine."""
    AWAKENING = "awakening"  # Grace period countdown at start of round
    PATROL = "patrol"        # Roaming/surveying when distant from player
    PURSUIT = "pursuit"      # Active A* pursuit of the player's position
    STUNNED = "stunned"      # Immobilized by player smoke/freeze ability


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Admissible and consistent heuristic for 4-directional grid topology.

    Args:
        a: (x, y) coordinates of the source node.
        b: (x, y) coordinates of the destination node.

    Returns:
        L1 taxicab distance |a[0] - b[0]| + |a[1] - b[1]|.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_path(maze, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Compute the optimal shortest path from start to goal via A* search.

    Maintains admissibility through Manhattan distance heuristic. Solvability
    is guaranteed on perfect CSP spanning-tree mazes.

    Args:
        maze: Maze object providing grid bounds and open neighbor traversal.
        start: Origin coordinate tuple (x, y).
        goal: Target coordinate tuple (x, y).

    Returns:
        Ordered list of coordinates [start, ..., goal], or [] if unreachable.
    """
    if start == goal:
        return [start]

    # Priority queue stores tuples of: (f_score, g_score, coordinates)
    open_heap: list[tuple[int, int, tuple[int, int]]] = [(manhattan(start, goal), 0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], int] = {start: 0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if current in closed:
            continue

        if current == goal:
            # Reconstruct optimal path backwards from goal via parent pointers
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed.add(current)

        for nb in maze.neighbors(*current):
            tentative_g = g + 1
            if tentative_g < g_score.get(nb, float("inf")):
                g_score[nb] = tentative_g
                came_from[nb] = current
                f = tentative_g + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, tentative_g, nb))

    return []  # Fallback for disconnected graphs (unreachable in valid CSP mazes)


class Hunter:
    """Hunter entity governed by a behavioral state machine with dynamic A* tracking.

    Implements:
      - Awakening countdown (grace period preventing round-start ambushes).
      - Proximity-based state transitions between PATROL and PURSUIT.
      - Tactical stun recovery duration.
      - Cadence-based movement intervals that scale fairly with difficulty.
    """

    def __init__(
        self,
        position: tuple[int, int],
        replan_interval: int = 20,
        move_interval: int = 28,
        awakening_duration_s: float = 0.0,
        fps: int = 30,
    ):
        self.position = position
        self.replan_interval = max(1, replan_interval)
        self.move_interval = max(1, move_interval)
        self.fps = fps

        # FSM and State Timers
        self.awakening_frames = int(awakening_duration_s * fps)
        self.state = HunterState.AWAKENING if self.awakening_frames > 0 else HunterState.PATROL
        self.stun_frames = 0
        self.detection_radius = 8  # Cells within which pursuit triggers
        self.patrol_target: tuple[int, int] | None = None

        self._frame = 0
        self._path: list[tuple[int, int]] = []
        self._nodes_expanded_last = 0

    @property
    def is_awakening(self) -> bool:
        return self.state == HunterState.AWAKENING

    @property
    def awakening_time_left_s(self) -> float:
        return max(0.0, self.awakening_frames / self.fps)

    @property
    def is_stunned(self) -> bool:
        return self.state == HunterState.STUNNED

    @property
    def stun_time_left_s(self) -> float:
        return max(0.0, self.stun_frames / self.fps)

    def set_difficulty(
        self,
        replan_interval: int,
        move_interval: int,
        awakening_duration_s: float = 0.0,
    ) -> None:
        """Calibrate cadence parameters per predicted skill classification."""
        self.replan_interval = max(1, replan_interval)
        self.move_interval = max(1, move_interval)
        self.awakening_frames = int(awakening_duration_s * self.fps)
        if self.awakening_frames > 0:
            self.state = HunterState.AWAKENING

    def stun(self, duration_s: float = 3.0) -> None:
        """Apply temporary tactical immobilization from player smoke ability."""
        self.stun_frames = max(self.stun_frames, int(duration_s * self.fps))
        self.state = HunterState.STUNNED

    def update(self, maze, player_position: tuple[int, int]) -> bool:
        """Advance the Hunter by one game tick.

        Args:
            maze: Active maze structure for pathing and obstacle validation.
            player_position: Current coordinates of the player target.

        Returns:
            True if the hunter changed grid coordinates on this frame, else False.
        """
        self._frame += 1
        moved = False

        # State 1: Awakening Grace Period
        if self.state == HunterState.AWAKENING:
            self.awakening_frames -= 1
            if self.awakening_frames <= 0:
                self.state = HunterState.PURSUIT
            return False

        # State 2: Stunned Immobilization
        if self.state == HunterState.STUNNED:
            self.stun_frames -= 1
            if self.stun_frames <= 0:
                self.state = HunterState.PURSUIT
            return False

        # State 3: Evaluate Patrol vs. Pursuit transitions
        dist_to_player = manhattan(self.position, player_position)
        if dist_to_player <= self.detection_radius:
            self.state = HunterState.PURSUIT
            active_move_interval = self.move_interval
            target_pos = player_position
        else:
            self.state = HunterState.PATROL
            # Patrol operates at a gentler cadence (30% slower) to grant breathing room
            active_move_interval = int(self.move_interval * 1.3)
            # In Patrol mode: navigate toward patrol waypoints rather than tracking the player
            if not self.patrol_target or self.position == self.patrol_target or len(self._path) <= 1:
                w, h = maze.width, maze.height
                waypoints = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, h // 2)]
                candidates = [c for c in waypoints if c != self.position]
                self.patrol_target = candidates[self._frame % len(candidates)] if candidates else maze.exit
            target_pos = self.patrol_target

        # Dynamic A* Path Re-planning
        if self._frame % self.replan_interval == 0 or len(self._path) <= 1:
            self._path = astar_path(maze, self.position, target_pos)

        # Discrete Grid Movement Step
        if self._frame % active_move_interval == 0 and len(self._path) > 1:
            self.position = self._path[1]
            self._path.pop(0)
            moved = True

        return moved

    def caught(self, player_position: tuple[int, int]) -> bool:
        """Check collision intersection with the player."""
        return self.position == player_position


# Backward compatibility alias
HunterEntity = Hunter
