"""
Master Class Game Engine & Entity State Management.

===============================================================================
THEORETICAL FOUNDATION & DESIGN PATTERNS:
===============================================================================
1. STATE MACHINE PATTERN:
   The round lifecycle is formalized as a discrete finite state machine:
       PLAYING -> { ESCAPED, CAUGHT, TIMEOUT }
   Terminal state transitions are irreversible within a single round.

2. ENTITY-COMPONENT SEPARATION (OOP):
   - PlayerEntity: Encapsulates player state, positional history, backtrack
     counting, decision telemetry, and active strategic abilities (Dash, Smoke, Radar).
   - HunterEntity: Encapsulated pursuer with behavioral FSM (Awakening, Patrol,
     Pursuit, Stunned) and dynamic A* trajectory planning.
   - RoundEngine: The mediator/orchestrator owning the spatial grid (Maze) and
     driving the frame execution tick.

3. STRATEGY PATTERN (PLAYER ACTIVE ABILITIES):
   Solves the asymmetric information handicap by providing active tactical counterplay:
   - DASH (Mobility): High-velocity 2-cell burst through open passages.
   - SMOKE FREEZE (Tactical Stun): Deploys an obscuring smoke trap that stuns
     the Hunter for 3.0s, giving critical time to escape dead ends.
   - SONAR RADAR (Spatial Recon): Pings the maze to reveal cardinal vector and
     Manhattan distance toward the exit.
===============================================================================
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from .astar import Hunter, HunterState, astar_path
from .maze_gen import Maze, generate_maze


class Outcome(Enum):
    """Terminal and active states of a single game round."""
    PLAYING = "playing"
    ESCAPED = "escaped"
    CAUGHT = "caught"
    TIMEOUT = "timeout"


@dataclass
class RoundConfig:
    """Configuration parameters for maze sizing, time limits, and cadences."""
    width: int = 14
    height: int = 14
    round_time_limit_s: float = 60.0
    hunter_replan_interval: int = 20   # Frames between A* path regenerations
    hunter_move_interval: int = 28     # Frames between discrete cell steps
    awakening_duration_s: float = 2.5  # Grace period before hunter begins moving
    smoke_charges: int = 1
    radar_charges: int = 2
    dash_cooldown_s: float = 8.0
    fps: int = 30
    seed: int | None = None
    route_guide: bool = False

    def __post_init__(self):
        if self.width < 2 or self.height < 2:
            raise ValueError("Maze width and height must be at least 2")
        if self.fps <= 0 or not math.isfinite(self.round_time_limit_s) or self.round_time_limit_s < 0:
            raise ValueError("FPS must be positive and the time limit must be finite and nonnegative")
        if min(self.hunter_replan_interval, self.hunter_move_interval) < 1:
            raise ValueError("Hunter intervals must be positive")
        if min(self.smoke_charges, self.radar_charges, self.dash_cooldown_s, self.awakening_duration_s) < 0:
            raise ValueError("Ability charges and durations cannot be negative")


class PlayerEntity:
    """Encapsulates player state, move verification, and tactical abilities."""

    def __init__(
        self,
        start_pos: tuple[int, int],
        fps: int = 30,
        smoke_charges: int = 1,
        radar_charges: int = 2,
        dash_cooldown_s: float = 8.0,
    ):
        self.position = start_pos
        self.fps = fps
        self.path_taken: list[tuple[int, int]] = [start_pos]
        self.backtracks = 0
        self.decision_times_s: list[float] = []
        self._last_decision_ts = time.monotonic()
        self.last_move_direction = "N"

        # Tactical Abilities Cooldowns & Charges
        self.dash_cooldown_max = int(dash_cooldown_s * fps)
        self.dash_cooldown_frames = 0
        self.smoke_charges = smoke_charges
        self.radar_charges = radar_charges
        self.radar_cooldown_max = int(4.0 * fps)
        self.radar_cooldown_frames = 0

    @property
    def can_dash(self) -> bool:
        return self.dash_cooldown_frames <= 0

    @property
    def dash_cooldown_remaining_s(self) -> float:
        return max(0.0, self.dash_cooldown_frames / self.fps)

    @property
    def can_smoke(self) -> bool:
        return self.smoke_charges > 0

    @property
    def can_radar(self) -> bool:
        return self.radar_charges > 0 and self.radar_cooldown_frames <= 0

    def update_cooldowns(self) -> None:
        """Decrement ability cooldown counters by one frame."""
        if self.dash_cooldown_frames > 0:
            self.dash_cooldown_frames -= 1
        if self.radar_cooldown_frames > 0:
            self.radar_cooldown_frames -= 1

    def record_move(self, new_pos: tuple[int, int], direction: str, timestamp: float | None = None) -> None:
        """Log legal position transition and evaluate backtrack heuristics."""
        now = time.monotonic() if timestamp is None else timestamp
        self.decision_times_s.append(now - self._last_decision_ts)
        self._last_decision_ts = now
        self.last_move_direction = direction

        # Backtrack: Stepping directly back to the cell occupied prior to the last step
        if len(self.path_taken) >= 2 and new_pos == self.path_taken[-2]:
            self.backtracks += 1

        self.position = new_pos
        self.path_taken.append(new_pos)


class RoundEngine:
    """Master game loop engine orchestrating environment, entities, and state transitions."""

    def __init__(self, config: RoundConfig):
        self.config = config
        self.elapsed_s = 0.0
        self._simulated = False
        self._tick_accumulator = 0.0
        self.maze: Maze = generate_maze(config.width, config.height, seed=config.seed)
        self.player = PlayerEntity(
            self.maze.start,
            fps=config.fps,
            smoke_charges=config.smoke_charges,
            radar_charges=config.radar_charges,
            dash_cooldown_s=config.dash_cooldown_s,
        )
        self.player._last_decision_ts = 0.0
        self.hunter = Hunter(
            position=self._far_corner_from_start(),
            replan_interval=config.hunter_replan_interval,
            move_interval=config.hunter_move_interval,
            awakening_duration_s=config.awakening_duration_s,
            fps=config.fps,
        )
        self.outcome = Outcome.PLAYING
        self.frame = 0
        self.start_time = time.monotonic()
        self._last_update_ts = self.start_time
        self.elapsed_s = 0.0

        # Pause system state
        self.is_paused = False
        self.total_paused_s = 0.0
        self._pause_start_ts = 0.0

        # Event notification buffer for UI banners (e.g. "DASH!", "HUNTER STUNNED!")
        self.recent_events: list[dict] = []

    def pause(self) -> None:
        """Pause the active round, freezing game timers and entity updates."""
        if self.outcome == Outcome.PLAYING and not self.is_paused:
            if not self._simulated:
                self.update()
            if self.outcome != Outcome.PLAYING:
                return
            self.is_paused = True
            self._pause_start_ts = time.monotonic()

    def resume(self) -> None:
        """Resume the paused round and compensate the elapsed timer."""
        if self.is_paused:
            self.is_paused = False
            self.total_paused_s += time.monotonic() - self._pause_start_ts
            self._last_update_ts = time.monotonic()
            self._pause_start_ts = 0.0

    def toggle_pause(self) -> bool:
        """Toggle paused state. Returns True if paused, False if active."""
        if self.is_paused:
            self.resume()
        else:
            self.pause()
        return self.is_paused

    # Backward-compatible attribute delegations
    @property
    def player_pos(self) -> tuple[int, int]:
        return self.player.position

    @player_pos.setter
    def player_pos(self, pos: tuple[int, int]) -> None:
        self.player.position = pos

    @property
    def path_taken(self) -> list[tuple[int, int]]:
        return self.player.path_taken

    @property
    def backtracks(self) -> int:
        return self.player.backtracks

    @property
    def decision_times_s(self) -> list[float]:
        return self.player.decision_times_s

    @property
    def _last_decision_ts(self) -> float:
        return self.player._last_decision_ts

    @_last_decision_ts.setter
    def _last_decision_ts(self, val: float) -> None:
        self.player._last_decision_ts = val

    def _far_corner_from_start(self) -> tuple[int, int]:
        """Spawn in a side corridor, rather than blocking the unique escape route."""
        route = astar_path(self.maze, self.maze.start, self.maze.exit)
        distances = {cell: 0 for cell in route}
        queue = deque(route)
        while queue:
            cell = queue.popleft()
            for neighbor in self.maze.neighbors(*cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)
        candidates = [cell for cell, distance in distances.items() if distance > 0]
        if not candidates:
            candidates = [cell for cell in route if cell not in (self.maze.start, self.maze.exit)]
        if not candidates:
            return self.maze.exit
        return max(candidates, key=lambda cell: (distances[cell], self.maze.shortest_path_len(self.maze.start, cell)))

    def _prepare_action(self) -> bool:
        if self.outcome != Outcome.PLAYING or self.is_paused:
            return False
        if not self._simulated:
            self.update()
        return self.outcome == Outcome.PLAYING

    def _record_move(self, new_pos, direction):
        self.player.record_move(new_pos, direction, timestamp=self.elapsed_s)

    def move_player(self, direction: str) -> bool:
        """Attempt discrete cardinal movement for the player entity.

        Args:
            direction: One of {'N', 'S', 'E', 'W'}.

        Returns:
            True if the passage was unblocked and position updated, False if wall blocked.
        """
        if not self._prepare_action() or direction not in self.maze.DIRS:
            return False

        x, y = self.player.position
        if self.maze.walls[y][x][direction]:
            return False  # Wall collision — move rejected

        dx, dy = self.maze.DIRS[direction]
        new_pos = (x + dx, y + dy)
        self._record_move(new_pos, direction)

        # Check terminal state conditions
        if self.player.position == self.maze.exit:
            self.outcome = Outcome.ESCAPED
        elif self.hunter.caught(self.player.position):
            self.outcome = Outcome.CAUGHT

        return True

    def use_ability(self, ability_name: str, direction: str | None = None) -> dict | bool:
        """Execute an active tactical ability for the player.

        Supported abilities:
          - 'dash': Rapid 2-cell advance in chosen (or last moved) direction.
          - 'smoke': Deploy smoke trap to stun Hunter for 3.0 seconds.
          - 'radar': Spatial echo revealing distance and orientation toward Exit.
        """
        if not self._prepare_action():
            return False

        if ability_name == "dash":
            if not self.player.can_dash:
                return False
            dash_dir = direction or self.player.last_move_direction
            x, y = self.player.position
            # Smart fallback: if chosen direction is walled off, pick first open direction
            if self.maze.walls[y][x].get(dash_dir, True):
                open_dirs = [d for d, walled in self.maze.walls[y][x].items() if not walled]
                if open_dirs:
                    dash_dir = open_dirs[0]

            moved_any = False
            # Dash moves up to 2 cells if paths are open
            for _ in range(2):
                x, y = self.player.position
                if self.maze.walls[y][x][dash_dir]:
                    break
                dx, dy = self.maze.DIRS[dash_dir]
                new_pos = (x + dx, y + dy)
                self._record_move(new_pos, dash_dir)
                moved_any = True
                if self.player.position == self.maze.exit:
                    self.outcome = Outcome.ESCAPED
                    break
                if self.hunter.caught(self.player.position):
                    self.outcome = Outcome.CAUGHT
                    break

            if moved_any:
                self.player.dash_cooldown_frames = self.player.dash_cooldown_max
                self.recent_events.append({"text": "BURST DASH!", "time": self.elapsed_s})
                return True
            return False

        elif ability_name == "smoke":
            if not self.player.can_smoke:
                return False
            self.player.smoke_charges -= 1
            self.hunter.stun(duration_s=3.0)
            self.recent_events.append({"text": "HUNTER STUNNED (3.0s)!", "time": self.elapsed_s})
            return True

        elif ability_name == "radar":
            if not self.player.can_radar:
                return False
            self.player.radar_charges -= 1
            self.player.radar_cooldown_frames = self.player.radar_cooldown_max

            dist_exit = self.maze.shortest_path_len(self.player.position, self.maze.exit)
            dist_hunter = self.maze.shortest_path_len(self.player.position, self.hunter.position)

            ex, ey = self.maze.exit
            px, py = self.player.position
            cardinal = []
            if ey < py: cardinal.append("NORTH")
            elif ey > py: cardinal.append("SOUTH")
            if ex > px: cardinal.append("EAST")
            elif ex < px: cardinal.append("WEST")
            heading = "-".join(cardinal) or "AT EXIT"

            # Compute next 8 steps along optimal route to exit for visual guidance
            guide_route = astar_path(self.maze, self.player.position, self.maze.exit)
            guide_path = guide_route[1:9] if len(guide_route) > 1 else []

            info = {
                "distance_to_exit": dist_exit,
                "distance_to_hunter": dist_hunter,
                "exit_heading": heading,
                "guide_path": guide_path,
            }
            self.recent_events.append({
                "text": f"RADAR: Exit is {dist_exit} cells {heading}",
                "time": self.elapsed_s,
            })
            return info

        return False

    def route_guide_path(self) -> list[tuple[int, int]]:
        """A short, updating trail teaches navigation without revealing the whole solution."""
        return astar_path(self.maze, self.player_pos, self.maze.exit)[1:7] if self.config.route_guide else []

    @property
    def route_progress(self) -> float:
        optimal = self.maze.shortest_path_len(self.maze.start, self.maze.exit)
        remaining = self.maze.shortest_path_len(self.player_pos, self.maze.exit)
        return max(0.0, min(1.0, 1.0 - remaining / max(1, optimal)))

    def update(self, simulated_dt: float | None = None) -> Outcome:
        """Advance a shared clock; entity ticks remain independent of rendering FPS."""
        if self.outcome != Outcome.PLAYING or self.is_paused:
            return self.outcome

        now = time.monotonic()
        if simulated_dt is not None:
            if not math.isfinite(simulated_dt) or simulated_dt < 0:
                raise ValueError("Simulation delta must be finite and nonnegative")
            self._simulated = True
            remaining = simulated_dt
        else:
            remaining = max(0.0, now - self._last_update_ts)
        self._last_update_ts = now
        tick_s = 1.0 / self.config.fps
        while remaining > 1e-10 and self.outcome == Outcome.PLAYING:
            step = min(remaining, tick_s - self._tick_accumulator,
                       max(0.0, self.config.round_time_limit_s - self.elapsed_s))
            self.elapsed_s += step
            self._tick_accumulator += step
            remaining -= step
            if self._tick_accumulator >= tick_s - 1e-10:
                self._tick_accumulator = 0.0
                self.frame += 1
                self.player.update_cooldowns()
                self.hunter.update(self.maze, self.player.position)
                if self.hunter.caught(self.player.position):
                    self.outcome = Outcome.CAUGHT
            if self.elapsed_s >= self.config.round_time_limit_s - 1e-10:
                self.elapsed_s = self.config.round_time_limit_s
                if self.outcome == Outcome.PLAYING:
                    self.outcome = Outcome.TIMEOUT

        if self.hunter.caught(self.player.position):
            self.outcome = Outcome.CAUGHT
        elif self.elapsed_s >= self.config.round_time_limit_s:
            self.outcome = Outcome.TIMEOUT

        # Prune older UI transient events (older than 2.5 seconds)
        self.recent_events = [e for e in self.recent_events if self.elapsed_s - e["time"] < 2.5]

        return self.outcome

    def win_probability_features(self) -> dict:
        """Live engineered feature vector for logistic regression win prediction."""
        dist_to_hunter = self.maze.shortest_path_len(self.player.position, self.hunter.position)
        dist_to_exit = self.maze.shortest_path_len(self.player.position, self.maze.exit)
        time_left_ratio = max(0.0, 1 - self.elapsed_s / self.config.round_time_limit_s) if self.config.round_time_limit_s else 0.0
        cells_travelled = max(1, len(self.player.path_taken) - 1)
        efficiency = (
            self.maze.shortest_path_len(self.maze.start, self.player.position) / cells_travelled
            if self.player.position != self.maze.start
            else 1.0
        )
        return {
            "dist_to_hunter": dist_to_hunter,
            "dist_to_exit": dist_to_exit,
            "time_left_ratio": time_left_ratio,
            "efficiency": efficiency,
        }
