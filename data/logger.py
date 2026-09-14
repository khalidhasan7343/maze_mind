"""
Round-by-Round SQLite Telemetry Ingestion & Persistence Engine.

===============================================================================
THEORETICAL FOUNDATION & DESIGN PRINCIPLES:
===============================================================================
1. CONTINUOUS TELEMETRY STREAMING & TRANSACTIONAL INTEGRITY (ACID):
   Every completed round (whether human or bot-simulated) is committed atomically
   to persistent SQLite storage. Transactions guarantee durability and consistency
   under rapid headless simulations or sudden window terminations.

2. SCHEMA AS A CANONICAL FEATURE CONTRACT:
   The relational schema defines the definitive feature contract between the game
   runtime engine (`game/`) and the statistical learning pipeline (`ml/`).
   Decoupling logging from training enables:
     - Asynchronous offline model re-training.
     - Longitudinal player skill tracking across sessions.
     - Cold-start bootstrapping from Monte Carlo simulations.

3. COGNITIVE TIME-SERIES SERIALIZATION:
   Per-step decision latencies [t_1, t_2, ..., t_N] are stored as JSON arrays,
   preserving the microsecond-level temporal distribution of player decisions
   for downstream variance and latency distribution analysis.
===============================================================================
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field

from contextlib import contextmanager

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "schema.sql")
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "maze_mind.db")


@dataclass
class RoundRecord:
    session_id: str
    player_tag: str
    is_bot: bool
    maze_width: int
    maze_height: int
    optimal_path_len: int
    dead_end_count: int
    player_path_len: int
    time_taken_s: float
    backtracks: int
    decision_times_s: list = field(default_factory=list)
    outcome: str = "timeout"          # 'escaped' | 'caught' | 'timeout'
    hunter_replan_ms: int = 0
    hunter_move_ms: int = 0
    skill_label: str | None = None

    @property
    def avg_decision_time_s(self) -> float:
        if not self.decision_times_s:
            return 0.0
        return sum(self.decision_times_s) / len(self.decision_times_s)


class RoundLogger:
    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with open(_SCHEMA_PATH, "r") as f:
            schema = f.read()
        with self._connect() as conn:
            conn.executescript(schema)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def log_round(self, record: RoundRecord) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO rounds (
                    session_id, player_tag, is_bot, maze_width, maze_height,
                    optimal_path_len, dead_end_count, player_path_len,
                    time_taken_s, backtracks, decision_times_s,
                    avg_decision_time_s, outcome, hunter_replan_ms,
                    hunter_move_ms, skill_label
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    record.player_tag,
                    int(record.is_bot),
                    record.maze_width,
                    record.maze_height,
                    record.optimal_path_len,
                    record.dead_end_count,
                    record.player_path_len,
                    record.time_taken_s,
                    record.backtracks,
                    json.dumps(record.decision_times_s),
                    record.avg_decision_time_s,
                    record.outcome,
                    record.hunter_replan_ms,
                    record.hunter_move_ms,
                    record.skill_label,
                ),
            )
            return cur.lastrowid

    def update_skill_label(self, round_id: int, skill_label: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE rounds SET skill_label = ? WHERE round_id = ?",
                (skill_label, round_id),
            )

    def fetch_all(self) -> list:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM rounds ORDER BY round_id ASC").fetchall()
            return [dict(r) for r in rows]

    def fetch_by_player(self, player_tag: str) -> list:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM rounds WHERE player_tag = ? ORDER BY round_id ASC",
                (player_tag,),
            ).fetchall()
            return [dict(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]

    def get_player_summary(self, player_tag: str | None = None) -> dict:
        """Aggregate career statistics and recent match logs for a player."""
        rounds = []
        if player_tag:
            rounds = self.fetch_by_player(player_tag)

        # Fallback to all human rounds or all rounds if tag has no direct rows
        if not rounds:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM rounds WHERE is_bot = 0 ORDER BY round_id ASC").fetchall()
                rounds = [dict(r) for r in rows]
            if not rounds:
                rounds = self.fetch_all()

        if not rounds:
            return {
                "total_rounds": 0,
                "escapes": 0,
                "caught": 0,
                "timeouts": 0,
                "win_rate_pct": 0.0,
                "best_time_s": None,
                "best_efficiency_pct": 0.0,
                "recent_rounds": [],
            }

        total = len(rounds)
        escapes = sum(1 for r in rounds if r.get("outcome") == "escaped")
        caught = sum(1 for r in rounds if r.get("outcome") == "caught")
        timeouts = sum(1 for r in rounds if r.get("outcome") == "timeout")
        escaped_rounds = [r for r in rounds if r.get("outcome") == "escaped"]
        best_time = min((r["time_taken_s"] for r in escaped_rounds), default=None)

        efficiencies = []
        for r in escaped_rounds:
            opt = r.get("optimal_path_len", 0)
            actual = max(1, r.get("player_path_len", 1))
            efficiencies.append(min(100.0, 100.0 * opt / actual))

        best_eff = max(efficiencies, default=0.0)

        return {
            "total_rounds": total,
            "escapes": escapes,
            "caught": caught,
            "timeouts": timeouts,
            "win_rate_pct": round(100.0 * escapes / total, 1) if total > 0 else 0.0,
            "best_time_s": round(best_time, 1) if best_time is not None else None,
            "best_efficiency_pct": round(best_eff, 1),
            "recent_rounds": list(reversed(rounds[-10:])),
        }



if __name__ == "__main__":
    logger = RoundLogger(db_path="/tmp/test_maze_mind.db")
    rid = logger.log_round(RoundRecord(
        session_id="test-session",
        player_tag="smoke-test",
        is_bot=True,
        maze_width=10,
        maze_height=10,
        optimal_path_len=18,
        dead_end_count=6,
        player_path_len=22,
        time_taken_s=12.3,
        backtracks=2,
        decision_times_s=[0.2, 0.3, 0.25],
        outcome="escaped",
        hunter_replan_ms=250,
        hunter_move_ms=300,
    ))
    print(f"Logged round_id={rid}, total rows={logger.count()}")
    os.remove("/tmp/test_maze_mind.db")
