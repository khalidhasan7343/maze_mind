# Maze Mind — Judge Guide

**Purpose:** understand the implementation, explain it in a presentation, demonstrate it, and answer viva questions. This guide describes the code and local artifacts inspected on **5 October 2026**. Model scores below come from saved artifacts dated **4 October 2026**; they are not newly retrained results.

Read sections 1–4 for the overall design, sections 5–10 for implementation and ML, and sections 11–15 for testing and presentation. [importants notes.txt](<importants notes.txt>) contains a shorter speaking script.

## 1. What is my project?

**Maze Mind is a desktop maze escape game with an AI hunter and a machine learning layer that analyzes gameplay.** The player must reach the exit before the hunter catches them or the timer expires. The application offers Classic and Isometric views, selectable difficulty, player abilities, persistent records, and offline model training.

The project combines these subjects:

| Subject | Application in this project |
| --- | --- |
| GUI and event handling | Pygame windows, keyboard/mouse input, menus, pause, results |
| Graph algorithms | DFS maze generation, BFS distances, A* hunter pathfinding |
| Software architecture | Separate presentation, game rules, session services, persistence, and ML |
| Databases | SQLite stores completed rounds and intermediate game snapshots |
| Supervised ML | KNN skill classification and logistic regression escape prediction |
| Unsupervised ML | K-Means playstyle clustering |
| Data analysis | Feature engineering, scaling, PCA, evaluation, report charts |
| Software testing | Automated rules, UI, database, ML, and regression tests |

A suitable opening statement:

> “My project is Maze Mind, a maze escape game that integrates classical AI search with machine learning. A* controls the hunter, while gameplay records support skill classification, playstyle analysis, and escape probability estimates. Two interfaces share the same game engine and SQLite database.”

## 2. Technology stack and why it is used

| Technology | Responsibility | Why it fits |
| --- | --- | --- |
| Python 3.12+ | Application language | One language for the game, database access, models, and tests |
| pygame-ce, imported as `pygame` | Desktop frontend | Handles graphics, events, window state, and drawing |
| Python classes/dataclasses | Backend services and configuration | Keep game rules separate from screen drawing |
| `sqlite3` and SQLite | Persistent storage | Embedded database with no separate server setup |
| NumPy | Numeric operations | Feature vectors, variance, model inputs |
| pandas | Tabular data processing | Converts saved rounds into training tables |
| scikit-learn | Models and evaluation | KNN, K-Means, PCA, logistic regression, splits, scaling |
| joblib | Model persistence | Saves and loads fitted model bundles |
| Matplotlib | Report generation | Exports charts for analysis and presentations |
| pytest | Automated checks | Verifies behavior after code changes |
| GitHub Actions | Continuous integration | Runs headless checks on pushes and pull requests |

[requirements.txt](requirements.txt) lists direct dependencies. [requirements.lock](requirements.lock) pins the validated environment. [pyproject.toml](pyproject.toml) defines packaging, the Python requirement, and the installed `maze-mind` command.

## 3. Architecture: how frontend, backend, database, and ML connect

### 3.1 The important distinction

This is a **desktop Python application**. Its frontend and backend are logical layers in the same process. It does not currently have a browser frontend, HTTP API, or separate backend server.

Therefore, “connecting the frontend to the backend” means importing classes and calling Python methods. “Connecting the backend to the database” means opening the local SQLite file through `sqlite3`.

```text
User: keyboard / mouse
          |
          v
main.py -> Home: choose Classic or Isometric
                     |
                     v
          Selected UI: choose level, start, render
                     |
          +----------+----------+
          |                     |
          v                     v
     RoundSession          RoundEngine
     difficulty            movement and walls
     player history        timer and abilities
     feature snapshots     hunter and outcomes
     finish/predict              |
          |                 Maze + BFS + A*
          v
     RoundLogger <-------> SQLite: db/maze_mind.db
          ^                 rounds + round_snapshots
          |
     Offline training: preprocessing / bot simulation
          |
          v
     ml/models/*.joblib
          |
          v
     Optional predictors -> skill, style, escape estimate
          |
          +-> next Adaptive round / Classic HUD / stored results

     Dashboard reads data/models -> report PNG files
     Tests exercise these layers in isolated environments
```

### 3.2 What is the frontend?

The frontend is the code under [game/ui](game/ui). It displays information and translates input into actions. It does not decide whether a player can pass through a wall; the engine makes that decision.

| Frontend module | Purpose |
| --- | --- |
| [home.py](game/ui/home.py) | Choose Classic or Isometric using previews |
| [pygame_ui.py](game/ui/pygame_ui.py) | Classic screen controller and game loop |
| [isometric_ui.py](game/ui/isometric_ui.py) | Isometric level menu, gameplay, pause, results |
| [isometric_render.py](game/ui/isometric_render.py) | Project grid cells into raised isometric geometry |
| [screens.py](game/ui/screens.py) | Classic menu, history, pause, and results screens |
| [rendering.py](game/ui/rendering.py) | Classic maze, actor, and exit drawing |
| [widgets.py](game/ui/widgets.py) | Reusable buttons and telemetry displays |
| [controls.py](game/ui/controls.py) | Keyboard directions, held movement, mouse helpers |
| [theme.py](game/ui/theme.py) | Shared appearance, layout, and screen-state definitions |
| [navigation.py](game/ui/navigation.py) | Home navigation result shared by both views |

### 3.3 What is the backend?

The main backend responsibilities are split between two classes:

**`RoundEngine`, in [engine.py](game/engine.py):** owns the current maze, player, hunter, elapsed time, abilities, movement rules, and outcome. It can run without a visible window, allowing bots and tests to use the same rules.

**`RoundSession`, in [session.py](game/session.py):** coordinates selected difficulty, player history, snapshots, logging, and optional predictions. Both interfaces use it, so view selection does not require two separate database or ML implementations.

[main.py](main.py) is the composition point: it parses commands, builds configuration, chooses the frontend, and passes the logger and player information into that frontend.

### 3.4 The actual connection points

| Connection | Code relationship |
| --- | --- |
| CLI → frontend | `cmd_play()` imports the selected view's `run_round()` |
| CLI → configuration | `RoundConfig(...)` receives maze size, timer, and seed |
| CLI → database service | Creates `RoundLogger()` after a view is selected |
| UI → session | Creates a `RoundSession` with configuration, player tag, and logger |
| Session → engine | `session.start(tier, ...)` creates a fresh `RoundEngine` |
| UI → movement | `engine.move_player(direction)` validates and applies movement |
| UI → running game | `engine.update()` advances time and hunter behavior |
| UI → snapshots | `session.capture_snapshot()` samples current game features |
| UI → completion | `session.finish()` saves a terminal round once |
| Session → persistence | `logger.log_round(record, snapshots)` inserts the result |
| Session → ML | Calls optional skill/style predictors on completed-round features |
| Classic UI → live ML | Uses the win predictor with `engine.win_probability_features()` |
| Training → SQLite | Reads eligible rounds or saved human snapshots through the logger |

These are ordinary method calls, not REST requests. No API URL, port, database password, or frontend environment variable is required.

## 4. One complete user journey

1. The user runs `python main.py play --tag faculty_demo`.
2. `main()` parses arguments and dispatches `cmd_play()`.
3. Home displays Classic and Isometric previews. Closing Home before choosing a view does not create a gameplay record or initialize the logger.
4. The chosen view opens its level menu. Back to Home, or Escape from that menu, returns `NavigationAction.HOME` to `cmd_play()`, which opens Home again.
5. The user chooses Recruit, Runner, Master, Adaptive, or Custom and starts.
6. `RoundSession.start()` creates a fresh configured engine, maze, player, hunter, and snapshot list.
7. Keyboard/mouse events become game actions. The engine validates walls and updates positions; the UI reads the resulting state and draws it.
8. Engine updates advance the timer, cooldowns, and hunter. The session periodically collects feature snapshots in memory.
9. The round reaches `ESCAPED`, `CAUGHT`, or `TIMEOUT`.
10. `finish()` creates a `RoundRecord` and stores it together with its collected snapshots.
11. If compatible models are available, completed-round skill and style are predicted and saved in separate prediction columns.
12. History and statistics refresh. A later Adaptive round can use recent skill estimates and loss assistance.

**Important:** round data and snapshots are written at completion; the database is not written on every drawing frame. An unfinished abandoned round is not saved as a completed game. Viewing results again does not insert a duplicate record.

The application state and the round outcome are different concepts. Menus, pause, and Home describe navigation. `PLAYING`, `ESCAPED`, `CAUGHT`, and `TIMEOUT` describe game results. A Home navigation action is not a win or loss.

## 5. Gameplay, algorithms, and the isometric view

### 5.1 Maze generation: randomized DFS

[maze_gen.py](game/maze_gen.py) represents each cell using north, south, east, and west walls. A wall value indicates whether that boundary is closed.

Generation starts with closed walls and uses randomized, iterative depth-first search:

1. Visit a cell and choose an unvisited neighboring cell.
2. Remove the wall between them in both cells.
3. Continue forward, using a stack to backtrack when no unvisited neighbor remains.
4. Finish when all cells are visited.

This builds a connected tree of corridors. Every cell is reachable, and the generated tree has one simple route between any two cells. The start is `(0, 0)` and the exit is the opposite corner.

The generator also checks route-length constraints. If a candidate is too short, it retries within a bounded number of attempts. Small-grid constraints are adjusted to what is feasible. A seed reproduces maze generation for debugging and demonstrations.

Some project comments call this a CSP maze. A precise explanation is **randomized DFS generation with connectivity and path-length constraints**, rather than a general-purpose constraint-satisfaction solver.

### 5.2 BFS: measuring shortest corridor distance

`Maze.shortest_path_len()` uses breadth-first search. All legal corridor steps have equal cost, so BFS gives the shortest number of steps.

It supplies the optimal start-to-exit distance, remaining distance, hunter detection distance, and live model distances. It also provides an independent reference for A* correctness tests. A coordinate distance can be small while corridor distance is large because walls force a detour.

### 5.3 A*: planning the hunter's route

[astar.py](game/astar.py) searches legal maze neighbors using a priority queue:

```text
f(n) = g(n) + h(n)
g(n): actual steps from the search start to cell n
h(n): Manhattan distance from n to the target
```

Manhattan distance is a lower bound for legal four-direction movement. Walls can increase the route length, so A* still checks corridors rather than moving directly through walls.

The search stores the best discovered cost and predecessor for each cell, then reconstructs a route. The hunter recalculates routes as required by its state, target movement, and configured cadence. An optimal path is to the target position used for that search; it does not predict all future player actions.

**A* is the hunter's search algorithm. It is not a trained ML model.** KNN can change the hunter's timing in Adaptive mode, but it does not replace A* route planning.

### 5.4 Hunter state and shared time

The hunter has awakening, patrol, pursuit, and stunned behavior. Classic play includes a grace period and corridor-distance detection. Smoke temporarily stuns the hunter and protects against contact while that stun applies.

In Isometric play, the round waits for the first **legal movement**. The timer and pursuit then begin together. The hunter continually pursues according to its movement interval.

This is a real-time chase: the hunter can continue moving after the player stops. It does not take exactly one step for every player keypress. Starting together and stepping at the same frequency are different behaviors.

The engine accumulates elapsed time and advances entities on fixed simulation ticks. The default is 30 ticks per second. Rendering speed does not define hunter speed. Tests and bots can pass simulated elapsed time to `update()` without waiting in real time. Pause freezes gameplay time, movement, and cooldown progression.

### 5.5 Player controls

| Input | Use |
| --- | --- |
| WASD / arrow keys | Move through open neighboring cells; held keys repeat |
| Mouse navigation | Select a legal adjacent cell where supported by the view |
| Shift | Dash, attempting additional movement subject to walls and cooldown |
| F | Smoke: temporarily stun the hunter |
| R | Radar: temporarily reveal a limited route segment |
| P / Escape during play | Pause |
| Back to Home / Escape in the level menu | Return to view selection |

Recruit also has a route guide to make learning and successful play easier. Ability charges and cooldowns vary by level. Losing window focus pauses play automatically.

### 5.6 Difficulty selection

| Level | Maze | Timer | Main purpose |
| --- | --- | --- | --- |
| Recruit | 8 × 8 | 120 seconds | Guided, slower chase, more ability charges |
| Runner | 12 × 12 | 75 seconds | Intermediate challenge |
| Master | 14 × 14 | 60 seconds | Faster hunter and fewer resources |
| Adaptive | Base configuration | Base timer, with assistance when needed | Adjust hunter timing using recent skill/history |
| Custom | CLI dimensions | CLI timer | Preserve explicitly supplied settings |

At the default tick rate, Recruit's hunter step interval is two seconds in Classic and at least three seconds in Isometric. Classic Recruit has an eight-second awakening period; Isometric starts pursuit on the first legal move instead.

Adaptive uses the median rank of recent valid skill estimates/labels to reduce abrupt changes. After two recent unsuccessful rounds, it adds practical assistance such as a route guide, slower hunter, extra smoke, and more time when necessary. Fixed Recruit/Runner/Master presets are not automatically replaced by the model's suggested difficulty.

### 5.7 How isometric “3D” works

[isometric_render.py](game/ui/isometric_render.py) draws a 2D grid as tilted diamonds with raised walls and block-like actors. Conceptually:

```text
screen_x = origin_x + (grid_x - grid_y) * half_width
screen_y = origin_y + (grid_x + grid_y) * half_width * 0.5
                         - height * half_width
```

Height shifts a point upward on screen. Polygon faces create depth, draw order handles overlaps, and interpolated positions smooth actor movement. Inverse projection helps translate a mouse click back into a grid cell.

The accurate term is **isometric / 2.5D rendering**. The logical world is still a two-dimensional maze; this is not a full 3D engine with a freely rotating camera or 3D physics.

Both views share game rules and storage. Their displays differ: the live win-probability meter and post-round style display are in Classic. Isometric currently emphasizes progress, timer, and hunter status; it still collects snapshots and stores completed-round predictions.

## 6. Database: connection, tables, and saved information

### 6.1 Connecting SQLite

[data/logger.py](data/logger.py) contains `RoundLogger`. Its default database is `db/maze_mind.db`, resolved relative to the project code.

When constructed, the logger:

1. Ensures the database directory exists.
2. Loads [db/schema.sql](db/schema.sql).
3. Creates missing tables and indexes.
4. Adds supported missing columns to older schemas while preserving records.

Its connection helper calls `sqlite3.connect(self.db_path)`, enables foreign keys, manages commit/rollback, and closes the connection afterward. Inserts use bound `?` parameters for data values.

There is no database service to start. The `.db` file is the persistent database; `schema.sql` is its structure definition. SQLite is already available through Python's standard library.

### 6.2 Relationship between the tables

```text
rounds                              round_snapshots
------------------------            ------------------------
round_id (primary key)   1 --------< round_id (foreign key)
one completed game                  snapshot_id (primary key)
final outcome                       several sampled game states
raw gameplay measurements           four live numeric features
labels / predictions
```

This is a **one-to-many relationship**. A round may have several snapshots. Their eventual win/loss label comes from the parent's outcome, avoiding repetition of a whole round record for every sampled state.

### 6.3 `rounds` fields

| Group | Fields and meaning |
| --- | --- |
| Identification | `round_id`: database ID; `session_id`: UUID for the round; `player_tag`: history grouping name; `is_bot`: synthetic or human |
| Maze | `maze_width`, `maze_height`, `optimal_path_len`, `dead_end_count` |
| Movement | `player_path_len`, `remaining_path_len`, `backtracks` |
| Time | `time_taken_s`, `decision_times_s` as JSON, `avg_decision_time_s` |
| Result | `outcome`: escaped, caught, or timeout |
| Hunter configuration | `hunter_replan_ms`, `hunter_move_ms` |
| Ground truth | `skill_label`, `label_source`: synthetic, verified, or unknown |
| Model outputs | `predicted_skill`, `predicted_style`, `model_version` |
| Compatibility/history | `telemetry_version`, `created_at` |

`player_tag` is not a password-protected account. `session_id` is not a web login session. This is a local single-player application with named history.

`round_snapshots` contains `snapshot_id`, `round_id`, `dist_to_hunter`, `dist_to_exit`, `time_left_ratio`, and `efficiency`.

### 6.4 What happens when a round finishes?

`RoundSession.finish()` gathers raw engine measurements into a `RoundRecord`. `log_round()` inserts the parent and collected snapshots in the same transaction, so a successful save contains both. Model predictions are then stored separately through `update_predictions()`.

This order keeps the completed telemetry record independent from optional model availability. The session's recorded `round_id` prevents repeated finalization from adding duplicate rounds.

### 6.5 Ground truth versus prediction

These are separate database concepts:

- **`skill_label`:** an independent training label supplied by a synthetic bot profile or an explicit reviewer.
- **`predicted_skill`:** the model's guess about a completed round.
- **`label_source`:** identifies where the independent label came from.

KNN training uses independently labelled rows, not its own predictions as new ground truth. Otherwise, errors could reinforce themselves.

To label an actually reviewed existing record:

```bash
python main.py label --round-id 12 --skill Beginner
```

Replace `12` and the skill with a real reviewed round. This command updates persistent data; it should reflect a reasoned assessment, not simply copy a prediction.

### 6.6 Reading records

```bash
python main.py records --tag faculty_demo
```

The CLI and UI use logger queries for totals, escapes/catches/timeouts, win rate, best escape time, efficiency, and recent matches. The database survives application restarts.

Example read-only SQL to show the relationship in a database viewer:

```sql
SELECT r.round_id, r.outcome, COUNT(s.snapshot_id) AS snapshot_count
FROM rounds AS r
LEFT JOIN round_snapshots AS s ON s.round_id = r.round_id
GROUP BY r.round_id, r.outcome
ORDER BY r.round_id DESC
LIMIT 10;
```

## 7. Data collection and feature engineering

### 7.1 Why synthetic bots are needed

A new project has little human data. [data/simulate.py](data/simulate.py) provides three predefined bot skill profiles with different decision behavior. They play using the same game engine and generate labelled synthetic examples.

```bash
python main.py simulate --rounds-per-profile 120
```

Three profiles mean this requests **360 rounds**. It appends to the database; it does not clear previous rounds. These are bot labels, not proof of how real human skill categories behave.

Human completed rounds are also stored. Human skill labels require independent review for supervised KNN training. Escape labels for logistic regression come directly from actual final game outcomes.

### 7.2 Cleaning the dataset

[data/preprocess.py](data/preprocess.py) converts logger rows into pandas DataFrames. The current training path filters to supported telemetry version 2, removes zero-movement and invalid records, checks outcomes and numeric validity, and derives features.

Legacy records can remain in history without being treated as compatible training examples. Telemetry version describes data semantics; model version identifies a saved fitted artifact.

### 7.3 Completed-round features

| Feature | Meaning |
| --- | --- |
| `time_taken_s` | How long the round lasted |
| `efficiency_ratio` | Net progress toward the exit divided by steps taken, clipped to 0–1 |
| `backtracks` | Immediate reversals in the player's path |
| `avg_decision_time_s` | Mean recorded time between movement decisions |
| `decision_time_variance` | Variation in those decision delays |
| `outcome_code` | escaped = 0, caught = 1, timeout = 2 |
| `maze_size` | Width multiplied by height |
| `dead_end_count` | Number of corridor dead ends |

Not every engineered feature is selected by every model. KNN selects six features; K-Means selects five.

Completed-round efficiency is:

```text
progress = max(0, optimal_start_to_exit_distance - remaining_exit_distance)
efficiency_ratio = min(1, progress / max(1, steps_taken))
```

For an escaped round, remaining distance is zero, so this becomes optimal route length divided by actual steps. Example: a 40-step optimal route completed in 50 steps has efficiency 0.8. For an unsuccessful round, remaining distance matters; failure is not automatically treated as perfect efficiency.

Decision timing is a gameplay measurement. It is not a validated psychological diagnosis or pure measurement of human intelligence.

`RoundSession.finish()` explicitly includes the calculated
`RoundRecord.avg_decision_time_s` when building predictor inputs, because
`asdict()` includes dataclass fields but omits properties. Skill and style
prediction therefore receive the same average that the logger stores.
Regression tests compare immediate predictor inputs with saved and batch
features for both view policies, including single-move and empty histories.

### 7.4 Live snapshot features

`RoundEngine.win_probability_features()` creates a different four-feature vector:

| Feature | Meaning |
| --- | --- |
| `dist_to_hunter` | Shortest legal corridor distance to the hunter |
| `dist_to_exit` | Shortest legal corridor distance to the exit |
| `time_left_ratio` | Remaining time divided by total time |
| `efficiency` | Shortest start-to-current distance divided by steps already taken |

At the starting cell, live efficiency uses 1.0. Live efficiency and completed-round efficiency have different definitions and should not be described as interchangeable.

The session captures an initial snapshot and then samples at intervals based on roughly one-sixth of the time limit. Early termination can produce fewer snapshots. Features describe the state before completion; the eventual result supplies the training target afterward.

### 7.5 Why scaling matters

Time can span tens of seconds while efficiency spans 0–1. Without scaling, a distance-based model could be dominated by numeric units rather than useful behavior differences.

```text
scaled_value = (value - training_minimum) / (training_maximum - training_minimum)
```

KNN puts `MinMaxScaler` inside its sklearn pipeline. Each evaluation split learns scaling from its training rows, preventing test information from setting the scale. K-Means saves scale parameters and reuses them for inference and report projections. Constant-range features are handled without division by zero.

The current logistic regression uses raw snapshot features. Scaling balances numeric ranges; it does not prove features have equal importance.

## 8. The three machine learning models

| Model | Learning type | Input | Output | Application |
| --- | --- | --- | --- | --- |
| KNN | Supervised multiclass classification | Completed-round behavior | Beginner / Intermediate / Expert | Skill estimate and later Adaptive hunter timing |
| K-Means | Unsupervised clustering | Completed-round behavior | Descriptive playstyle group | Analysis, saved style, Classic results |
| Logistic regression | Supervised binary classification | Mid-round state | Probability of eventual escape | Classic live meter |

### 8.1 KNN: how skill is classified

Implementation: [ml/knn_skill.py](ml/knn_skill.py).

1. Read valid, independently labelled current telemetry.
2. Select time, efficiency, backtracks, mean decision time, decision variance, and outcome code.
3. Make a stratified holdout split, approximately 80% training / 20% testing.
4. Fit a scaling-plus-KNN pipeline on training rows.
5. Evaluate held-out accuracy and stratified cross-validation, up to five folds depending on available class counts.
6. Refit on all eligible labelled rows after evaluation and save the deployment bundle.

The default is **k = 5** with Euclidean distance. A new scaled observation gets the majority class of its nearest training neighbors.

Outcome is allowed as a feature here because classification happens after a round ends. It is not included as a live input to the escape-probability model.

Recent skill history maps to hunter movement/replanning intervals and Classic awakening time when Adaptive is selected. The maze route algorithm remains A*. Adaptation happens when starting the next round; the trained KNN parameters do not update after each move.

### 8.2 K-Means: how playstyles are grouped

Implementation: [ml/kmeans_strategy.py](ml/kmeans_strategy.py).

K-Means uses the five behavioral features above, excluding outcome code. It normalizes them and repeatedly assigns observations to their nearest centroid, then recalculates centroid means.

The default is **three clusters**, `random_state=42`, and ten initializations. Numeric cluster IDs are assigned descriptive names using cluster statistics. Current saved names include aggressive rusher, balanced runner, and cautious explorer. These names are heuristics, not independently verified personality categories.

An elbow scan records inertia for candidate cluster counts 2–7. Inertia is within-cluster squared distance, not classification accuracy. The default cluster count is configured; the code does not automatically prove or select the optimal count from the elbow.

**PCA** projects the five scaled features into two components for the cluster scatter chart. Clustering itself still uses the five selected features. PCA changes the visualization; it does not control the hunter.

### 8.3 Logistic regression: how escape probability is estimated

Implementation: [ml/logreg_winprob.py](ml/logreg_winprob.py).

Its target is `won = 1` for escape and `won = 0` for caught or timeout. It learns from mid-round snapshots labelled using the eventual round outcome.

```text
z = weight_1 * dist_to_hunter
  + weight_2 * dist_to_exit
  + weight_3 * time_left_ratio
  + weight_4 * efficiency
  + bias

P(eventual escape) = 1 / (1 + exp(-z))
```

The fitted coefficient signs come from the dataset; they should not be assumed solely from intuition. Despite its name, this usage of logistic regression is probabilistic classification.

By default, training generates a fresh bot snapshot dataset, with 60 rounds per profile, and adds eligible saved human snapshots when present. It is not limited to the bot final-round rows already in SQLite.

All snapshots from one round stay on the same side of the train/test split. Otherwise, nearly related states from one game could appear in both partitions and inflate evaluation results. The model is evaluated on held-out rounds, then refitted on all eligible snapshots for deployment.

The Classic HUD uses the fitted model's `predict_proba()` output. It is an estimate, not a guarantee of success. Its reported discrimination scores do not establish that every displayed percentage is calibrated for human players or every difficulty/view.

### 8.4 Model files and inference

Training creates:

```text
ml/models/knn_skill.joblib
ml/models/kmeans_strategy.joblib
ml/models/logreg_winprob.joblib
```

A bundle contains the fitted model, feature list, version metadata, and relevant evaluation/scaling/PCA information. [ml/artifacts.py](ml/artifacts.py) defines the feature-version contract and bundle validation.

Prediction functions cache loaded bundles to avoid repeated disk loading. The session wraps optional predictors, so missing or incompatible supported artifacts can disable an estimate while the game remains playable. Training is an explicit offline command, not continuous online learning or reinforcement learning.

**Saved models are not the database.** The database stores observations; the `.joblib` files store fitted models and their metadata. New records do not change an existing fitted model until training is run again.

## 9. Actual ML performance and how to explain it honestly

These results were read from local saved bundles, not invented or inferred from the number of passing software tests.

| Saved result | Value | Evaluation context |
| --- | --- | --- |
| KNN held-out accuracy | **100%** | 72 synthetic test rounds; zero human test rounds |
| KNN mean cross-validation accuracy | **100%** | Saved stratified CV result on eligible labelled data |
| Logistic regression accuracy | **86.54%** | 104 synthetic held-out snapshots |
| Logistic regression ROC AUC | **0.858** | Same saved held-out evaluation |
| Logistic regression round split | **144 train / 36 test rounds** | Entire rounds kept separate |
| Logistic regression human test samples | **0** | Human performance not measured by this saved result |
| K-Means clusters | **3** | Unsupervised groups |
| K-Means inertia | **18.7695** | Compactness on its scaled training data |
| PCA first component variance | **91.08%** | Variance retained in the first projection axis |
| PCA second component variance | **5.99%** | Variance retained in the second projection axis |
| PCA first two combined | **97.07%** | Visualization retains most variance in this fitted dataset |

Accuracy is the fraction of correctly classified held-out examples. ROC AUC measures how well probabilities rank escapes above unsuccessful rounds across thresholds; 0.858 is not “85.8% calibrated probability.” PCA variance retention is not prediction accuracy.

Saved elbow values:

| Number of clusters | Inertia |
| --- | --- |
| 2 | 44.3389 |
| 3 | 18.7695 |
| 4 | 12.1425 |
| 5 | 8.7378 |
| 6 | 7.3405 |
| 7 | 6.1374 |

There is a substantial drop from two to three groups, with smaller reductions afterward. This supports discussing a simple three-group representation; it does not prove that three human playstyles exist.

The live database contained **368 completed rounds**, comprising **360 synthetic rounds** and **8 human rounds with unknown label source**, plus **23 stored snapshots**, when inspected. These current totals are not the saved artifact's training/evaluation counts. They change as more games are played.

Use this wording in a presentation:

> “The saved KNN model achieved 100% accuracy on its synthetic held-out dataset. The escape model achieved approximately 86.54% accuracy and 0.858 ROC AUC. These results show the pipeline works on the generated bot domain. They do not establish real-player accuracy, because the saved held-out evaluations contain no human examples.”

Why caution is needed:

- Bot profiles are predefined and can be easier to distinguish than actual human behavior.
- Human evaluation needs enough independent reviewed examples and, ideally, separation by player.
- Current view and difficulty behavior can differ from bot training configurations.
- Refit models use all eligible data after evaluation, while saved metrics describe the earlier held-out experiment.
- K-Means inertia depends on the dataset and scaling; comparing it across different datasets without context is misleading.

## 10. Dashboard and project file map

[dashboard/report_plots.py](dashboard/report_plots.py) exports four analysis figures through:

```bash
python main.py dashboard
```

The images are written to `dashboard/figures/`. Open them from that folder
when preparing your slides.

| Figure | What to explain |
| --- | --- |
| `elbow_curve.png` | Inertia against candidate cluster count |
| `pca_clusters.png` | Two-component view of scaled behavioral observations and groups |
| `astar_benchmark.png` | Measured search timing for the benchmark configurations |
| `skill_breakdown.png` | Distribution of labelled skill categories in the data |

The dashboard is a report generator, not a separate web frontend. Benchmark timings depend on machine and workload. Category counts are not an accuracy score. Local screenshots such as Home and Isometric previews illustrate the UI rather than ML evaluation.

### Remaining files and directories

| File or folder | Responsibility |
| --- | --- |
| [main.py](main.py) | CLI dispatch and service wiring |
| [game/engine.py](game/engine.py) | Configuration, actors, rules, timing, outcomes, live features |
| [game/session.py](game/session.py) | Shared lifecycle, presets, assistance, predictors, completion |
| [game/maze_gen.py](game/maze_gen.py) | Maze representation, generation, neighbors, BFS distances |
| [game/astar.py](game/astar.py) | Path search and hunter behavior |
| [data/logger.py](data/logger.py) | Database connections, records, labels, summaries, migrations |
| [data/preprocess.py](data/preprocess.py) | Cleaning, derived features, numeric scaling |
| [data/simulate.py](data/simulate.py) | Bot profiles and synthetic completed-round collection |
| [ml/knn_skill.py](ml/knn_skill.py) | Skill training, inference, history aggregation, cadence mapping |
| [ml/kmeans_strategy.py](ml/kmeans_strategy.py) | Clusters, names, elbow scan, PCA, inference |
| [ml/logreg_winprob.py](ml/logreg_winprob.py) | Snapshot generation, grouped evaluation, live probability |
| [ml/artifacts.py](ml/artifacts.py) | Bundle metadata and compatibility validation |
| [db/schema.sql](db/schema.sql) | Tables, fields, indexes, parent/snapshot relationship |
| [README.md](README.md) | Setup and contributor/gameplay documentation |
| [pytest.ini](pytest.ini) | pytest configuration |
| [.github/workflows/tests.yml](.github/workflows/tests.yml) | Automated CI workflow |
| [.gitignore](.gitignore) | Excludes generated/local artifacts from version control |
| Package `__init__.py` files | Identify/describe Python packages; they are not separate application servers |
| `.venv/` | Local interpreter and installed dependencies, not authored game logic |
| `ml/models/`, local `.db`, report figures | Generated artifacts; they can need rebuilding on another checkout |

The frontend module map appears in section 3.2, and every test module is described in section 11.

## 11. What is the use of the `tests` folder?

Tests are executable checks of expected behavior. They support development and demonstration confidence: if movement, database storage, or navigation changes, tests can detect broken behavior before a manual presentation.

They are not part of the player's normal game loop and do not automatically make an ML model accurate. **Software correctness tests and statistical model evaluation answer different questions.**

| Test file | What it checks |
| --- | --- |
| [conftest.py](tests/conftest.py) | Shared headless setup, temporary data/model fixtures, reusable test resources |
| [test_maze_and_astar.py](tests/test_maze_and_astar.py) | Maze connectivity, reproducible seeds, small grids, A* paths, hunter behavior |
| [test_engine.py](tests/test_engine.py) | Legal/illegal movement, abilities, pause, escape, catch, timeout |
| [test_data_pipeline.py](tests/test_data_pipeline.py) | SQLite round trips, JSON timing, preprocessing, scaling, records |
| [test_ml_models.py](tests/test_ml_models.py) | Training/inference behavior and model feature contracts |
| [test_regressions.py](tests/test_regressions.py) | Previously fragile behavior: migrations, label separation, split/scaling leakage, timing, cache switching, one-time saving, live/database feature parity |
| [test_cli.py](tests/test_cli.py) | CLI configuration and explicit label handling |
| [test_ui_flow.py](tests/test_ui_flow.py) | Classic menus, pause, history, results, logging flow |
| [test_isometric.py](tests/test_isometric.py) | Projection, first-move start, pursuit, pause, smoke, results, launch behavior |
| [test_home.py](tests/test_home.py) | View selection, cancel, Back to Home, cleanup, preserved configuration |
| [test_playability.py](tests/test_playability.py) | Held movement, safe smoke, spawning, guidance, assistance, seeded Recruit playability |
| [__init__.py](tests/__init__.py) | Test package marker/documentation |

### Fixtures and temporary resources

A pytest fixture prepares data shared by tests. For example, a fixture can create a temporary database, populate synthetic records, train small temporary models, and return their paths to the test.

This lets tests check real training and prediction without overwriting the player's live database or the main saved artifacts. Headless SDL display/audio drivers allow Pygame screens and events to be tested without an interactive desktop window. Matplotlib's Agg backend lets plots render without a graphical display.

Some assertions compare independent expected behavior, such as A* route length against BFS or checking that opening results twice does not save twice. Parametrized tests exercise multiple seeds or configurations, so the number of collected cases is greater than the number of test functions.

### Running and interpreting checks

```bash
python -m pytest -q
```

On 5 October 2026, the local run completed with **123 passed, 1 warning**, in approximately 21 seconds. This includes six new feature-parity cases covering both view policies. The warning was joblib falling back to the logical core count after physical-core detection failed; the run still passed.

Passing these checks supports the tested behaviors. It does not mean every possible bug is excluded, all human players will win, or an ML score has improved.

CI is configured for Python **3.12 and 3.14** on Ubuntu, installs locked dependencies, sets headless drivers, runs pytest, and checks CLI help on pushes and pull requests. The local result above is not a claim that a new remote CI run was performed.

## 12. Setup, connecting the layers, and demonstration commands

### 12.1 First setup on another machine

Run commands from the project root. On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python main.py --help
```

Use a Python installation that meets the 3.12+ requirement. On Windows, activate with `.venv\Scripts\activate` instead.

For a fresh checkout without data/models:

```bash
python main.py simulate --rounds-per-profile 120
python main.py train
python main.py dashboard
python main.py play --tag faculty_demo
```

Simulation appends records. Training replaces the default saved model artifacts. The scores generated on that machine/data may differ from section 9. Perform this preparation before a presentation; it need not be repeated before every game.

The game itself can run without trained models. Model outputs then become unavailable; maze generation, movement, and hunter search remain operational.

### 12.2 How the layers are connected after setup

There is no manual network connection step. Running `play` follows the wiring in `main.py`; selecting a view creates the logger and session. The logger initializes SQLite, and prediction code loads existing model bundles when used.

To use a separate database while developing a Python service/test, the logger accepts a path:

```python
from data.logger import RoundLogger

logger = RoundLogger(db_path="db/demo_records.db")
```

This creates/initializes that file when executed. The ordinary CLI uses its default database; it does not currently offer a `--db-path` switch.

### 12.3 Recommended live demonstration

1. Start Home with `python main.py play --tag faculty_demo`.
2. Explain the two view previews; open a level menu and use Back to Home.
3. Select Isometric and Recruit. Show that the timer/chase waits for the first legal move.
4. Move using the guide, show smoke or radar, and pause/resume. Aim to escape; a caught/timeout result also demonstrates persistence.
5. After a completed round, show records with `python main.py records --tag faculty_demo`.
6. Open Classic using `python main.py play --view classic --tag faculty_demo`; show the live probability meter if models are available.
7. Show report figures and explain the saved scores with their synthetic-data limitations.
8. Run `python -m pytest -q` to demonstrate automated verification.

To reproduce a particular maze in a demonstration:

```bash
python main.py play --view isometric --difficulty Recruit --tag faculty_demo --seed 42
```

A seed reproduces maze generation, not the player's future actions or a guaranteed result. Recruit is the recommended demonstration level because it provides guidance and a slower hunter.

For a custom configuration:

```bash
python main.py play --difficulty Custom --width 9 --height 7 --time-limit 45 --seed 42
```

Use Custom when demonstrating CLI dimensions and timer; fixed presets intentionally supply their own values.

## 13. Current limitations and sensible improvements

Explain what the implementation supports, then identify work needed to strengthen it:

1. **Human ML validation:** collect consented real-player data, independently review skill labels, and evaluate on separate players. Saved synthetic scores are not a substitute.
2. **View/difficulty coverage:** train/evaluate with representative Classic and Isometric configurations. First-move pursuit and assistance can change the data distribution.
3. **Probability calibration:** assess reliability of displayed probabilities, not only accuracy and AUC.
4. **Presentation consistency:** add the live probability/style display to Isometric if both views should expose the same analytics.
5. **Data/model traceability:** save more dataset/configuration identifiers with artifacts so evaluation can be reproduced after more records are added.
6. **Accessibility and enjoyment:** add configurable controls, stronger color/accessibility options, and human playtesting across ability levels.

The current application is local single-player. A future browser frontend would require a new network interface and architecture; it is not how today's Pygame frontend connects to the backend.

## 14. Faculty viva questions and suggested answers

**Q1. What is the main contribution?**  
An integrated playable system: classical maze/search algorithms, persistent telemetry, three ML tasks, two interfaces, and automated verification share the same core services.

**Q2. Where is the frontend and where is the backend?**  
The frontend is `game/ui`. The engine/session are the logical backend. They communicate through Python method calls in one desktop process.

**Q3. How do you connect the database?**  
`RoundLogger` opens the local SQLite file using `sqlite3`, initializes the schema, and performs transactional queries. No separate database server is needed.

**Q4. Which AI catches the player?**  
A* plans legal hunter routes. Hunter state and movement intervals decide when it follows them. The ML models do not directly choose its next corridor cell.

**Q5. Why use both BFS and A*?**  
BFS measures shortest unweighted corridor distances. A* uses a goal-directed heuristic for route planning. Comparing their results also supports correctness tests.

**Q6. What does machine learning actually change?**  
KNN estimates completed-round skill and informs later Adaptive timing. K-Means describes playstyle. Logistic regression estimates escape probability in the Classic HUD.

**Q7. Does the model learn while I move?**  
The game performs inference and collects data. Fitted models change only when the training command runs. History-based assistance can change settings without retraining.

**Q8. What is a snapshot?**  
A small numeric description of a running game: hunter distance, exit distance, remaining time ratio, and live efficiency. Its final result is known after the parent round ends.

**Q9. How do you prevent data leakage?**  
KNN learns scaling inside each evaluation training split. Logistic regression keeps all snapshots of a round together. Independent labels are stored separately from model predictions.

**Q10. Does 100% KNN accuracy mean every real player is classified correctly?**  
No. The saved held-out result is synthetic-only. Real-player accuracy needs independent human evaluation. The corrected feature handoff and its regression tests establish input consistency, not human classification accuracy.

**Q11. What is K-Means accuracy?**  
The current model has no supervised accuracy score. We discuss inertia, elbow behavior, and interpretability. PCA variance is a visualization measure.

**Q12. Is the isometric interface true 3D?**  
It uses 2.5D projection and raised polygon faces over a 2D grid. It is not a full 3D simulation.

**Q13. Why have a tests folder?**  
To verify rules, data persistence, models, UI navigation, and regressions automatically. Tests support reliability but do not replace human playtesting or statistical evaluation.

**Q14. What happens without model files?**  
Supported missing/incompatible optional predictors disable those estimates. Core maze play and A* still work. Training restores the artifacts when enough eligible data exists.

**Q15. Do you store every frame?**  
No. The session samples intermediate features in memory and stores them with a completed round. It does not commit a database row on every frame.

**Q16. How does the Back button work?**  
Both level menus return a shared Home navigation action. The CLI loop reopens Home and preserves configuration and the logger. Navigation is separate from round outcomes.

**Q17. Is this reinforcement learning?**  
No. It uses classical search, supervised learning, unsupervised clustering, and history-based settings. There is no reward-trained policy controlling the hunter.

**Q18. How can you improve the project next?**  
Evaluate independently labelled human data across views, assess probability calibration, extend feature-contract coverage, and run accessibility/playability studies.

## 15. Presentation structure and code walkthrough

For a seven-slide presentation:

| Slide | Content |
| --- | --- |
| 1 | Problem, objective, screenshot, project title |
| 2 | Architecture diagram: UI → engine/session → logger → SQLite → ML |
| 3 | DFS, BFS, A*, hunter states, isometric rendering |
| 4 | Database tables and example telemetry/features |
| 5 | Three ML tasks and how their outputs are used |
| 6 | Saved scores, synthetic/human distinction, report charts, 123 passing tests |
| 7 | Live demo, present limitations, next improvements |

For a source walkthrough, open files in this order:

1. `main.py`: show command dispatch and view selection.
2. `game/ui/home.py` and a selected UI: show input/rendering and Back navigation.
3. `game/session.py`: show `start()`, snapshots, and `finish()`.
4. `game/engine.py`: show movement, update clock, outcome, live features.
5. `game/maze_gen.py` and `game/astar.py`: show graph/search responsibilities.
6. `data/logger.py` and `db/schema.sql`: show the actual persistence connection.
7. `data/preprocess.py` and the three model modules: show training versus inference.
8. `tests/conftest.py` and one meaningful test: show isolation and verification.

Useful final statement:

> “Maze Mind demonstrates how a game interface, search-based AI, a relational database, and statistical models can work together. The architecture lets both views share the same rules and data pipeline. My next priorities are validating the models with independently reviewed human gameplay and evaluating probability calibration across views and difficulties.”
