-- Maze Mind round-by-round log.
-- One row per completed round. Raw, unnormalised — preprocessing.py
-- turns this into model-ready feature vectors.

CREATE TABLE IF NOT EXISTS rounds (
    round_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          TEXT NOT NULL,
    player_tag          TEXT,                -- human name or bot profile name
    is_bot              INTEGER NOT NULL DEFAULT 0,
    maze_width          INTEGER NOT NULL,
    maze_height         INTEGER NOT NULL,
    optimal_path_len    INTEGER NOT NULL,    -- BFS shortest path, in cells
    dead_end_count      INTEGER NOT NULL,
    player_path_len     INTEGER NOT NULL,    -- cells actually travelled
    time_taken_s        REAL NOT NULL,
    backtracks          INTEGER NOT NULL,
    decision_times_s    TEXT,                -- JSON list of per-move think times
    avg_decision_time_s REAL NOT NULL,
    outcome             TEXT NOT NULL,       -- 'escaped' | 'caught' | 'timeout'
    hunter_replan_ms    INTEGER NOT NULL,
    hunter_move_ms      INTEGER NOT NULL,
    skill_label         TEXT,                -- Beginner/Intermediate/Expert, if known
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_rounds_session ON rounds(session_id);
CREATE INDEX IF NOT EXISTS idx_rounds_outcome ON rounds(outcome);
