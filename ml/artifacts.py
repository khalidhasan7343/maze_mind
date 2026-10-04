"""Versioned artifacts shared by training and live inference."""

from datetime import datetime, timezone

FEATURE_VERSION = 2


def metadata():
    return {"feature_version": FEATURE_VERSION,
            "model_version": datetime.now(timezone.utc).isoformat()}


def validate_bundle(bundle):
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model uses older telemetry; run python main.py simulate and python main.py train")
    return bundle
