
import numpy as np


def clean_signal(signal):
    """Remove invalid numerical values from a signal."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal[np.isfinite(signal)]

    if len(signal) == 0:
        raise ValueError("Signal has no valid samples.")

    return signal


def create_windows(signal, window_size=4000, step_size=4000):
    """Split a signal into non-overlapping or overlapping windows."""
    signal = clean_signal(signal)

    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step sizes must be positive.")

    windows = []

    for start in range(0, len(signal) - window_size + 1, step_size):
        windows.append(signal[start:start + window_size])

    return windows
