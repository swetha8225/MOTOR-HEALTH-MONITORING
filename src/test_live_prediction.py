
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "results" / "motor_health.db"
MODEL_FILE = ROOT / "results" / "models" / "independent_bearing_model.joblib"

WINDOW_SIZE = 80


def calculate_features(signal):
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        raise ValueError("No valid vibration samples found.")

    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": float(rms),
        "peak": float(peak),
        "peak_to_peak": float(np.ptp(signal)),
        "kurtosis": float(kurtosis(signal)),
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
    }


def main():
    model = joblib.load(MODEL_FILE)

    with sqlite3.connect(DB_FILE) as conn:
        df = pd.read_sql_query(
            """
            SELECT ax
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(WINDOW_SIZE,),
        )

    df = df.iloc[::-1].reset_index(drop=True)

    if len(df) < WINDOW_SIZE:
        print(f"Need {WINDOW_SIZE} readings; found {len(df)}.")
        return

    signal = df["ax"].to_numpy(dtype=np.float64)

    if not np.isfinite(signal).all():
        print("Invalid sensor readings found.")
        return

    # Use the original signal, without removing its mean,
    # to match the current training feature definitions.
    features = pd.DataFrame([calculate_features(signal)])

    print("\nExtracted features:")
    print(features.to_string(index=False))

    prediction = model.predict(features)[0]
    print("\nExperimental model prediction:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        print("\nModel class probabilities:")

        for label, probability in zip(model.classes_, probabilities):
            print(f"{label:20s}: {probability:.2%}")

    print(
        "\nWARNING: The model was trained on Paderborn bearing data. "
        "Its predictions are not validated for this motor or sensor."
    )


if __name__ == "__main__":
    main()
