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
from dataclasses import dataclass, field
from enum import Enum

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

    def record_move(self, new_pos: tuple[int, int], direction: str) -> None:
        """Log legal position transition and evaluate backtrack heuristics."""
        now = time.monotonic()
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
        self.maze: Maze = generate_maze(config.width, config.height, seed=config.seed)
        self.player = PlayerEntity(
            self.maze.start,
            fps=config.fps,
            smoke_charges=config.smoke_charges,
            radar_charges=config.radar_charges,
            dash_cooldown_s=config.dash_cooldown_s,
        )
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
            self.is_paused = True
            self._pause_start_ts = time.monotonic()

    def resume(self) -> None:
        """Resume the paused round and compensate the elapsed timer."""
        if self.is_paused:
            self.is_paused = False
            self.total_paused_s += time.monotonic() - self._pause_start_ts
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
        """Place Hunter in an opposite quadrant away from both Start and Exit."""
        sx, sy = self.maze.start
        w, h = self.maze.width, self.maze.height
        corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
        corners = [c for c in corners if c != self.maze.exit and c != (sx, sy)]
        return corners[0] if corners else (w - 1, h - 1)

    def move_player(self, direction: str) -> bool:
        """Attempt discrete cardinal movement for the player entity.

        Args:
            direction: One of {'N', 'S', 'E', 'W'}.

        Returns:
            True if the passage was unblocked and position updated, False if wall blocked.
        """
        if self.outcome != Outcome.PLAYING:
            return False

        x, y = self.player.position
        if self.maze.walls[y][x][direction]:
            return False  # Wall collision — move rejected

        dx, dy = self.maze.DIRS[direction]
        new_pos = (x + dx, y + dy)
        self.player.record_move(new_pos, direction)

        # Check terminal state conditions
        if self.player.position == self.maze.exit:
            self.outcome = Outcome.ESCAPED
        elif self.player.position == self.hunter.position:
            self.outcome = Outcome.CAUGHT

        return True

    def use_ability(self, ability_name: str, direction: str | None = None) -> dict | bool:
        """Execute an active tactical ability for the player.

        Supported abilities:
          - 'dash': Rapid 2-cell advance in chosen (or last moved) direction.
          - 'smoke': Deploy smoke trap to stun Hunter for 3.0 seconds.
          - 'radar': Spatial echo revealing distance and orientation toward Exit.
        """
        if self.outcome != Outcome.PLAYING:
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
                self.player.record_move(new_pos, dash_dir)
                moved_any = True
                if self.player.position == self.maze.exit:
                    self.outcome = Outcome.ESCAPED
                    break
                if self.player.position == self.hunter.position:
                    self.outcome = Outcome.CAUGHT
                    break

            if moved_any:
                self.player.dash_cooldown_frames = self.player.dash_cooldown_max
                self.recent_events.append({"text": "BURST DASH!", "time": time.monotonic()})
                return True
            return False

        elif ability_name == "smoke":
            if not self.player.can_smoke:
                return False
            self.player.smoke_charges -= 1
            self.hunter.stun(duration_s=3.0)
            self.recent_events.append({"text": "HUNTER STUNNED (3.0s)!", "time": time.monotonic()})
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
                "time": time.monotonic(),
            })
            return info

        return False

    def update(self, simulated_dt: float | None = None) -> Outcome:
        """Advance simulation tick: update entity states, timers, and collisions."""
        if self.outcome != Outcome.PLAYING or self.is_paused:
            return self.outcome

        self.frame += 1
        if simulated_dt is not None:
            self.elapsed_s += simulated_dt
        else:
            self.elapsed_s = time.monotonic() - self.start_time - self.total_paused_s

        # Update entity timers & cooldowns
        self.player.update_cooldowns()
        self.hunter.update(self.maze, self.player.position)

        # Collision & Timeout evaluations
        if self.hunter.caught(self.player.position):
            self.outcome = Outcome.CAUGHT
        elif self.elapsed_s >= self.config.round_time_limit_s:
            self.outcome = Outcome.TIMEOUT

        # Prune older UI transient events (older than 2.5 seconds)
        now = time.monotonic()
        self.recent_events = [e for e in self.recent_events if now - e["time"] < 2.5]

        return self.outcome

    def win_probability_features(self) -> dict:
        """Live engineered feature vector for logistic regression win prediction."""
        dist_to_hunter = self.maze.shortest_path_len(self.player.position, self.hunter.position)
        dist_to_exit = self.maze.shortest_path_len(self.player.position, self.maze.exit)
        time_left_ratio = max(0.0, 1 - self.elapsed_s / self.config.round_time_limit_s)
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
