import os

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture(scope="session")
def trained_models(tmp_path_factory):
    """Actual training artifacts in isolation, independent of user model files."""
    from data.simulate import generate_dataset
    from ml.knn_skill import train as train_knn
    from ml.kmeans_strategy import train as train_kmeans
    from ml.logreg_winprob import collect_snapshots, train as train_logreg

    root = tmp_path_factory.mktemp("models")
    db = str(root / "rounds.db")
    generate_dataset(n_per_profile=15, db_path=db)
    paths = {name: str(root / f"{name}.joblib") for name in ["knn", "kmeans", "logreg"]}
    train_knn(db_path=db, model_path=paths["knn"])
    train_kmeans(db_path=db, model_path=paths["kmeans"])
    snapshots = collect_snapshots(n_rounds_per_profile=20)
    train_logreg(model_path=paths["logreg"], snapshot_frame=snapshots)
    return {**paths, "db": db, "snapshots": snapshots}
