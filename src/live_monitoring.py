from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.motor_feature_utils import FEATURES


@dataclass(frozen=True)
class LiveSnapshot:
    captured_at: str
    reading: dict[str, float]
    features: dict[str, float]
    prediction: str
    probabilities: dict[str, float] | None
    window_start_ms: float
    window_end_ms: float


def load_latest_live_snapshot(
    recording_file: Path,
    feature_file: Path,
    model: Any,
) -> LiveSnapshot:
    if not recording_file.exists():
        raise FileNotFoundError(f"Recording file not found: {recording_file}")
    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")
    if feature_file.stat().st_mtime_ns < recording_file.stat().st_mtime_ns:
        raise ValueError(
            "The latest recording has no matching complete feature window. "
            "At least 100 valid samples are required."
        )

    recording = pd.read_csv(recording_file)
    reading_columns = ["time_ms", "ax", "ay", "az", "total_accel"]
    missing_readings = set(reading_columns) - set(recording.columns)
    if missing_readings:
        raise ValueError(
            f"Recording is missing required columns: "
            f"{', '.join(sorted(missing_readings))}"
        )

    for column in reading_columns:
        recording[column] = pd.to_numeric(recording[column], errors="coerce")
    valid_readings = np.isfinite(
        recording[reading_columns].to_numpy(dtype=float)
    ).all(axis=1)
    recording = recording.loc[valid_readings]
    if recording.empty:
        raise ValueError("The recording contains no valid sensor rows.")

    features = pd.read_csv(feature_file)
    window_columns = FEATURES + ["start_time_ms", "end_time_ms"]
    missing_features = set(window_columns) - set(features.columns)
    if missing_features:
        raise ValueError(
            f"Feature file is missing required columns: "
            f"{', '.join(sorted(missing_features))}"
        )

    for column in window_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    valid_windows = np.isfinite(
        features[window_columns].to_numpy(dtype=float)
    ).all(axis=1)
    features = features.loc[valid_windows]
    if features.empty:
        raise ValueError("The feature file contains no complete valid windows.")

    latest_reading = recording.iloc[-1]
    latest_window = features.iloc[-1]
    if latest_window["end_time_ms"] > latest_reading["time_ms"]:
        raise ValueError(
            "The latest feature window does not match the current recording."
        )

    feature_values = {
        name: float(latest_window[name]) for name in FEATURES
    }
    model_input = pd.DataFrame([feature_values], columns=FEATURES)
    prediction = str(model.predict(model_input)[0])

    probabilities = None
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(model_input)[0]
        probabilities = {
            str(label): float(score)
            for label, score in zip(model.classes_, scores)
        }

    captured_at = datetime.fromtimestamp(
        recording_file.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")

    return LiveSnapshot(
        captured_at=captured_at,
        reading={
            name: float(latest_reading[name])
            for name in reading_columns
        },
        features=feature_values,
        prediction=prediction,
        probabilities=probabilities,
        window_start_ms=float(latest_window["start_time_ms"]),
        window_end_ms=float(latest_window["end_time_ms"]),
    )
