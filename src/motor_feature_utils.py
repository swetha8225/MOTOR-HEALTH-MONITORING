
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
]


def calculate_features(signal):
    signal = np.asarray(signal, dtype=float)
    signal = signal[np.isfinite(signal)]

    if len(signal) < 2:
        raise ValueError("At least two valid samples are required.")

    rms = float(np.sqrt(np.mean(signal ** 2)))
    peak = float(np.max(np.abs(signal)))

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.ptp(signal)),
        "kurtosis": float(kurtosis(signal)),
        "crest_factor": peak / rms if rms > 0 else 0.0,
    }


def calculate_window_features(df, axis="az"):
    if axis not in df.columns:
        raise ValueError(f"Missing vibration column: {axis}")

    return calculate_features(df[axis].to_numpy())
