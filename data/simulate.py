r"""
Monte Carlo Synthetic Gameplay Simulation & Cold-Start Data Ingestion.

===============================================================================
THEORETICAL FOUNDATION & LECTURE CONCEPTS:
===============================================================================
1. COLD-START INGESTION & SYNTHETIC BOOTSTRAPPING:
   Supervised classification (KNN) and unsupervised clustering (K-Means) require
   empirical behavioral distributions before live human players generate logs.
   This headless simulation generates representative telemetry across distinct
   synthetic behavioral archetypes by executing automated bot agents against
   the true underlying `RoundEngine` and physics loop.

2. EPSILON-GREEDY AGENT POLICIES (MARKOV DECISION PROCESS):
   Each bot profile formalizes an epsilon-greedy policy over candidate actions A(s):
     - With probability (1 - epsilon) = greediness:
       Select action a* = argmin_{a} d_{A*}(T(s, a), exit)
       (Optimal exploitation along the true A* shortest path).
     - With probability epsilon = mistake_rate:
       Select random legal exploratory action a in A(s) \ {a*}
       (Simulates hesitation, perceptual blind spots, and dead-end detours).

3. STOCHASTIC DECISION LATENCY MODELING:
   Cognitive decision latencies delta_t are sampled from bounded uniform intervals
   [t_min, t_max] reflecting cognitive processing speed:
     - Expert: Rapid recognition-primed decisions (30ms - 120ms).
     - Intermediate: Deliberate scanning cadence (80ms - 300ms).
     - Novice: Hesitant visual search (150ms - 600ms).

4. UNIFIED LOGGING CONTRACT:
   Both synthetic simulation and interactive Pygame sessions log into the
   identical SQLite schema. Synthetic behavior still differs from human play;
   source flags allow separate evaluation on real human telemetry.
===============================================================================
"""

from __future__ import annotations

import random
import sys
import time
import uuid

sys.path.append("..")

from game.engine import RoundEngine, RoundConfig, Outcome  # noqa: E402
from data.logger import RoundLogger, RoundRecord  # noqa: E402

DIRS = ["N", "S", "E", "W"]


class BotProfile:
    """A bot picks a direction each tick. `greediness` in [0,1] is the
    chance it moves along the true shortest path to the exit rather than
    a random legal direction — this is what separates skill classes."""

    def __init__(self, name: str, greediness: float, think_time_range: tuple,
                 mistake_rate: float):
        self.name = name
        self.greediness = greediness
        self.think_time_range = think_time_range
        self.mistake_rate = mistake_rate


PROFILES = [
    BotProfile("novice_bot", greediness=0.25, think_time_range=(0.15, 0.6), mistake_rate=0.35),
    BotProfile("intermediate_bot", greediness=0.6, think_time_range=(0.08, 0.3), mistake_rate=0.15),
    BotProfile("expert_bot", greediness=0.92, think_time_range=(0.03, 0.12), mistake_rate=0.03),
]

SKILL_FOR_PROFILE = {
    "novice_bot": "Beginner",
    "intermediate_bot": "Intermediate",
    "expert_bot": "Expert",
}


def _greedy_direction(engine: RoundEngine) -> str | None:
    """One step along the current BFS-shortest path to the exit."""
    maze = engine.maze
    from game.astar import astar_path
    path = astar_path(maze, engine.player_pos, maze.exit)
    if len(path) < 2:
        return None
    nx, ny = path[1]
    x, y = engine.player_pos
    dx, dy = nx - x, ny - y
    for d, (ddx, ddy) in maze.DIRS.items():
        if (ddx, ddy) == (dx, dy):
            return d
    return None


def _legal_directions(engine: RoundEngine) -> list:
    x, y = engine.player_pos
    return [d for d in DIRS if not engine.maze.walls[y][x][d]]


def run_bot_round(profile: BotProfile, config: RoundConfig, rng: random.Random) -> RoundRecord:
    for engine in iter_bot_round(profile, config, rng):
        pass

    optimal = engine.maze.shortest_path_len(engine.maze.start, engine.maze.exit)
    return RoundRecord(
        session_id=str(uuid.uuid4()),
        player_tag=profile.name,
        is_bot=True,
        maze_width=config.width,
        maze_height=config.height,
        optimal_path_len=optimal,
        dead_end_count=engine.maze.dead_end_count(),
        player_path_len=len(engine.path_taken) - 1,
        time_taken_s=round(min(config.round_time_limit_s, max(0.1, engine.elapsed_s)), 3),
        backtracks=engine.backtracks,
        decision_times_s=engine.decision_times_s,
        outcome=engine.outcome.value,
        hunter_replan_ms=int(1000 * config.hunter_replan_interval / config.fps),
        hunter_move_ms=int(1000 * config.hunter_move_interval / config.fps),
        skill_label=SKILL_FOR_PROFILE[profile.name],
        label_source="synthetic",
        remaining_path_len=engine.maze.shortest_path_len(engine.player_pos, engine.maze.exit),
    )


def iter_bot_round(profile: BotProfile, config: RoundConfig, rng: random.Random):
    """Yield shared-clock states; the hunter keeps moving while the bot thinks."""
    engine = RoundEngine(config)
    engine.update(simulated_dt=0.0)
    next_decision = rng.uniform(*profile.think_time_range)
    yield engine
    while engine.outcome == Outcome.PLAYING:
        engine.update(simulated_dt=1.0 / config.fps)
        if engine.outcome == Outcome.PLAYING and engine.elapsed_s + 1e-10 >= next_decision:
            legal = _legal_directions(engine)
            if not legal:
                engine.outcome = Outcome.TIMEOUT
            else:
                if rng.random() < profile.mistake_rate:
                    direction = rng.choice(legal)
                else:
                    greedy = _greedy_direction(engine)
                    direction = greedy if greedy and rng.random() < profile.greediness else rng.choice(legal)
                engine.move_player(direction)
                next_decision = engine.elapsed_s + rng.uniform(*profile.think_time_range)
        yield engine


def generate_dataset(n_per_profile: int = 120, db_path: str | None = None, seed: int = 0) -> int:
    if n_per_profile < 1:
        raise ValueError("rounds-per-profile must be positive")
    logger = RoundLogger(db_path=db_path) if db_path else RoundLogger()
    rng = random.Random(seed)
    total = 0
    for profile in PROFILES:
        for _ in range(n_per_profile):
            width = rng.choice([8, 10, 12, 14])
            height = rng.choice([8, 10, 12, 14])
            cfg = RoundConfig(
                width=width, height=height,
                round_time_limit_s=45.0,
                hunter_replan_interval=rng.choice([10, 15, 20]),
                hunter_move_interval=rng.choice([15, 20, 25]),
                seed=rng.randint(0, 1_000_000),
            )
            record = run_bot_round(profile, cfg, rng)
            logger.log_round(record)
            total += 1
    return total


if __name__ == "__main__":
    t0 = time.time()
    n = generate_dataset(n_per_profile=120)
    print(f"Logged {n} synthetic rounds in {time.time() - t0:.1f}s")
