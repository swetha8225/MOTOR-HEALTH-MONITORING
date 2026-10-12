
import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis
from pathlib import Path


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
        "crest_factor": float(peak / rms) if rms > 0 else 0.0
    }


def main():
    folder = Path("data/paderborn")
    files = sorted(folder.rglob("*.mat"))

    if not files:
        raise FileNotFoundError("No Paderborn MAT files found.")

    file = files[0]
    mat = loadmat(file, squeeze_me=True, struct_as_record=False)

    name = next(k for k in mat if not k.startswith("__"))
    data = mat[name]

    # Y[6] is the measured vibration_1 channel.
    vibration = np.asarray(data.Y[6].Data, dtype=np.float64)

    features = calculate_features(vibration)

    print("=" * 50)
    print("PADERBORN VIBRATION FEATURES")
    print("=" * 50)
    print("File:", file.name)
    print("Channel:", data.Y[6].Name)
    print("Raster:", data.Y[6].Raster)
    print("Samples:", len(vibration))

    for feature, value in features.items():
        print(f"{feature:16s}: {value:.6f}")


if __name__ == "__main__":
    main()
