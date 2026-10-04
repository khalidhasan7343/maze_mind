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
    label_source: str = "unknown"
    predicted_skill: str | None = None
    predicted_style: str | None = None
    model_version: str | None = None
    remaining_path_len: int | None = None
    telemetry_version: int = 2

    @property
    def avg_decision_time_s(self) -> float:
        if not self.decision_times_s:
            return 0.0
        return sum(self.decision_times_s) / len(self.decision_times_s)


class RoundLogger:
    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with open(_SCHEMA_PATH, "r") as f:
            schema = f.read()
        with self._connect() as conn:
            conn.executescript(schema)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(rounds)")}
            additions = {
                "label_source": "TEXT NOT NULL DEFAULT 'unknown'",
                "predicted_skill": "TEXT", "predicted_style": "TEXT",
                "model_version": "TEXT", "remaining_path_len": "INTEGER",
                "telemetry_version": "INTEGER NOT NULL DEFAULT 1",
            }
            for name, definition in additions.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE rounds ADD COLUMN {name} {definition}")
            conn.execute("UPDATE rounds SET label_source = 'synthetic' WHERE is_bot = 1 AND skill_label IS NOT NULL AND label_source = 'unknown'")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def log_round(self, record: RoundRecord, snapshots: list[dict] | None = None) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO rounds (
                    session_id, player_tag, is_bot, maze_width, maze_height,
                    optimal_path_len, dead_end_count, player_path_len,
                    time_taken_s, backtracks, decision_times_s,
                    avg_decision_time_s, outcome, hunter_replan_ms,
                    hunter_move_ms, skill_label, label_source, predicted_skill,
                    predicted_style, model_version, remaining_path_len, telemetry_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    record.label_source,
                    record.predicted_skill,
                    record.predicted_style,
                    record.model_version,
                    record.remaining_path_len,
                    record.telemetry_version,
                ),
            )
            if snapshots:
                conn.executemany(
                    "INSERT INTO round_snapshots (round_id, dist_to_hunter, dist_to_exit, time_left_ratio, efficiency) VALUES (?, ?, ?, ?, ?)",
                    [(cur.lastrowid, s["dist_to_hunter"], s["dist_to_exit"], s["time_left_ratio"], s["efficiency"]) for s in snapshots],
                )
            return cur.lastrowid

    def update_skill_label(self, round_id: int, skill_label: str) -> None:
        """Set a verified ground-truth label; predictions use update_predictions."""
        if skill_label not in {"Beginner", "Intermediate", "Expert"}:
            raise ValueError("Unknown skill label")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE rounds SET skill_label = ?, label_source = 'verified' WHERE round_id = ?",
                (skill_label, round_id),
            )
            if not cursor.rowcount:
                raise ValueError(f"Round {round_id} does not exist")

    def update_predictions(self, round_id: int, skill: str | None, style: str | None, model_version: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE rounds SET predicted_skill = ?, predicted_style = ?, model_version = ? WHERE round_id = ?",
                (skill, style, model_version, round_id),
            )

    def log_snapshots(self, round_id: int, snapshots: list[dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO round_snapshots (round_id, dist_to_hunter, dist_to_exit, time_left_ratio, efficiency) VALUES (?, ?, ?, ?, ?)",
                [(round_id, s["dist_to_hunter"], s["dist_to_exit"], s["time_left_ratio"], s["efficiency"]) for s in snapshots],
            )

    def fetch_snapshots(self) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT s.*, r.is_bot, CASE WHEN r.outcome = 'escaped' THEN 1 ELSE 0 END AS won
                FROM round_snapshots s JOIN rounds r ON r.round_id = s.round_id
                WHERE r.telemetry_version = 2 ORDER BY s.snapshot_id
            """).fetchall()
            result = [dict(r) for r in rows]
            for row in result:
                row["round_id"] = f"logged-{row['round_id']}"
            return result

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
        if player_tag is not None:
            rounds = self.fetch_by_player(player_tag)

        # Omitting the tag explicitly requests global human statistics.
        if player_tag is None:
            with self._connect() as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("SELECT * FROM rounds WHERE is_bot = 0 ORDER BY round_id ASC").fetchall()
                rounds = [dict(r) for r in rows]

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
