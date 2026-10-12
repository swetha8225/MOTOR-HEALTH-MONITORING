
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "results" / "motor_off.csv"
WINDOW_SIZE = 80


def main():
    if not DATA_FILE.exists():
        print(f"Recording not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    if "ax" not in df.columns:
        print("CSV must contain an 'ax' column.")
        return

    if len(df) < WINDOW_SIZE:
        print(f"Need {WINDOW_SIZE} readings; found {len(df)}.")
        return

    raw_signal = df["ax"].tail(WINDOW_SIZE).to_numpy(dtype=float)

    if not np.isfinite(raw_signal).all():
        print("Invalid sensor values found.")
        return

    signal = raw_signal - np.mean(raw_signal)

    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    peak_to_peak = np.ptp(signal)
    kurt = kurtosis(signal, fisher=False, bias=False)
    crest_factor = peak / rms if rms > 0 else 0.0

    features = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "kurtosis": kurt,
        "crest_factor": crest_factor,
    }

    print(f"\nRecording: {DATA_FILE.name}")
    print("Vibration features (ax, m/s²) - DC offset removed")
    print("-" * 50)

    for name, value in features.items():
        print(f"{name:15s}: {value:.6f}")

    print("\nVibration signal check")
    print("-" * 50)
    print(f"Number of samples  : {len(signal)}")
    print(f"Minimum vibration  : {np.min(signal):.6f} m/s²")
    print(f"Maximum vibration  : {np.max(signal):.6f} m/s²")
    print(f"Average vibration  : {np.mean(np.abs(signal)):.6f} m/s²")


if __name__ == "__main__":
    main()
