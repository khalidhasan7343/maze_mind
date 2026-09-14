# Maze Mind

AI Maze Runner with a Learning Layer — race the hunter, log every move,
let the data teach you.

One game that touches every lecture topic: a playable pygame maze runner
wired to a real ML pipeline. The player races an AI hunter through a
CSP-generated maze; every round is logged to SQLite; a separate ML
pipeline turns those logs into skill scores, playstyle clusters, and a
live win-probability meter; the predicted skill class feeds back into
the hunter's difficulty. The loop closes — this is integration, not two
projects glued together.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Bootstrap training data (bot-simulated rounds — real play adds more later)
python main.py simulate --rounds-per-profile 120

# 2. Train all three models
python main.py train

# 3. Generate the report figures
python main.py dashboard

# 4. Play for real (round is logged automatically; difficulty adapts
#    to your predicted skill class from the second round on)
python main.py play --tag yourname
```

Run the test suite any time with:

```bash
pytest -q
```

## Project layout

```
game/
  maze_gen.py       CSP maze generation (backtracking search)
  astar.py          A* hunter AI (dynamic re-planning)
  engine.py         Round state machine (PLAYING/ESCAPED/CAUGHT/TIMEOUT)
  ui/pygame_ui.py   Rendering, input, HUD (timer, win-prob meter, skill tag)
data/
  logger.py         SQLite round logging
  preprocess.py      Feature engineering + min-max normalisation
  simulate.py        Bot-driven synthetic data generator (3 skill profiles)
ml/
  knn_skill.py        KNN skill classifier -> difficulty knobs
  kmeans_strategy.py  K-Means playstyle clustering + elbow + PCA
  logreg_winprob.py   Logistic regression live win-probability
db/
  schema.sql          Table definition
  maze_mind.db        Created on first run (gitignored)
dashboard/
  report_plots.py     Elbow curve, PCA scatter, A* benchmarks, skill breakdown
tests/                pytest suite (maze/A*/engine/data pipeline)
main.py                CLI entry point (play / simulate / train / dashboard)
```

## Lecture-to-feature map

| Course topic | Where it lives |
|---|---|
| Constraint satisfaction (variables/domains/constraints/backtracking) | `game/maze_gen.py` — maze generation |
| Informed search (A*, admissible heuristics) | `game/astar.py` — hunter AI |
| Supervised learning — classification | `ml/knn_skill.py` — KNN skill class |
| Supervised learning — regression/probabilistic output | `ml/logreg_winprob.py` — sigmoid win probability |
| Unsupervised learning — clustering | `ml/kmeans_strategy.py` — K-Means + elbow method |
| Dimensionality reduction | `ml/kmeans_strategy.py` — PCA for the cluster figure |
| Data engineering / normalisation | `data/preprocess.py` — min-max scaling, why it matters for distance-based models |
| Databases | `db/schema.sql`, `data/logger.py` — SQLite round log |
| Software engineering — package separation, tests | `game/`, `data/`, `ml/`, `db/`, `dashboard/` + `tests/` |

## Notes for the report

- **CSP framing**: `game/maze_gen.py`'s module docstring states the
  variables/domains/constraints explicitly and explains why the
  randomised-DFS carve *is* backtracking search over that CSP.
- **Why normalise**: `data/preprocess.py` docstring — KNN and K-Means are
  distance-based, so an unnormalised feature with a big numeric range
  would dominate Euclidean distance.
- **Difficulty without cheating**: `ml/knn_skill.py`'s `DIFFICULTY_BY_SKILL`
  only changes the hunter's re-plan/move cadence, never the search
  algorithm itself.
- Report figures are regenerated any time with `python main.py dashboard`
  into `dashboard/figures/`.
