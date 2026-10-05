"""Define model provenance and the accepted feature version. Shared metadata and validation
prevent training and live prediction from silently using different telemetry
definitions.

Versioned artifacts shared by training and live inference.
"""

from datetime import datetime, timezone

FEATURE_VERSION = 2


def metadata():
    """Stamp saved artifacts with feature version and training time so inference and
    records can identify which model contract produced a prediction.
    """
    return {"feature_version": FEATURE_VERSION,
            "model_version": datetime.now(timezone.utc).isoformat()}


def validate_bundle(bundle):
    """Reject artifacts using an older feature contract so the game cannot silently apply a
    model trained on incompatible telemetry.
    """
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise ValueError("Model uses older telemetry; run python main.py simulate and python main.py train")
    return bundle
