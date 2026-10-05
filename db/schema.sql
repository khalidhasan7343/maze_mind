-- Purpose: define the persistent telemetry contract shared by human sessions and bots.
-- One rounds row represents a completed game. Store raw values so data/preprocess.py
-- can engineer features without losing the original measurements or decision timings.
-- data/logger.py applies this schema and adds missing legacy columns automatically.

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
    label_source        TEXT NOT NULL DEFAULT 'unknown', -- synthetic / verified / unknown
    predicted_skill     TEXT,
    predicted_style     TEXT,
    model_version       TEXT,
    remaining_path_len  INTEGER,
    telemetry_version   INTEGER NOT NULL DEFAULT 2,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Index common session, outcome, and player-history queries without changing stored data.
CREATE INDEX IF NOT EXISTS idx_rounds_session ON rounds(session_id);
CREATE INDEX IF NOT EXISTS idx_rounds_outcome ON rounds(outcome);
CREATE INDEX IF NOT EXISTS idx_rounds_player ON rounds(player_tag, round_id);

-- Mid-round states inherit their eventual outcome from the parent rounds record.
-- This separate table supports escape-probability training without repeating full records.
CREATE TABLE IF NOT EXISTS round_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id INTEGER NOT NULL REFERENCES rounds(round_id),
    dist_to_hunter REAL NOT NULL,
    dist_to_exit REAL NOT NULL,
    time_left_ratio REAL NOT NULL,
    efficiency REAL NOT NULL
);
-- Speed up joining feature snapshots to the completed round that labels them.
CREATE INDEX IF NOT EXISTS idx_snapshots_round ON round_snapshots(round_id);
