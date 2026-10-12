
import time
from pathlib import Path

import numpy as np
import pandas as pd
import serial
from scipy.stats import kurtosis

# ESP32 settings
PORT = "COM8"
BAUD_RATE = 115200
DURATION_SECONDS = 30
WINDOW_SIZE = 100

# Project output paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_DIR / "results"

RECORDING_FILE = RESULTS_DIR / "latest_motor_recording.csv"
FEATURE_FILE = RESULTS_DIR / "latest_motor_features.csv"

COLUMNS = [
    "time_ms", "ax", "ay", "az", "total_accel", "acs_adc"
]


def calculate_features(signal):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 2 or not np.isfinite(signal).all():
        raise ValueError("Signal contains insufficient or invalid samples.")

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


def collect_data():
    rows = []
    print(f"Connecting to ESP32 on {PORT} at {BAUD_RATE} baud...")

    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=2) as esp:
            time.sleep(2)
            start = time.time()
            print(f"Collecting for {DURATION_SECONDS} seconds...")
            print("Showing received serial lines below:")

            while time.time() - start < DURATION_SECONDS:
                line = esp.readline().decode(
                    "utf-8", errors="replace"
                ).strip()

                if not line:
                    continue

                print("ESP32:", repr(line))

                # Ignore CSV headers and status messages.
                if line.lower().startswith("time_ms"):
                    continue

                values = line.split(",")

                if len(values) != len(COLUMNS):
                    continue

                try:
                    row = [float(value) for value in values]
                except ValueError:
                    continue

                if not np.isfinite(row).all():
                    continue

                rows.append(row)

    except serial.SerialException as error:
        print(f"\nSerial connection failed: {error}")
        print("Check the USB connection, COM port, and Serial Monitor.")
        return None

    if not rows:
        print("\nNo valid six-column numeric readings received.")
        print("Check the ESP32 lines printed above and the firmware format.")
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    recording = pd.DataFrame(rows, columns=COLUMNS)
    recording.to_csv(RECORDING_FILE, index=False)

    print(f"\nCollected {len(recording)} readings.")
    print(f"Recording saved: {RECORDING_FILE}")

    return recording


def extract_features(recording):
    results = []

    for start in range(0, len(recording) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = recording.iloc[start:start + WINDOW_SIZE]
        signal = window["az"].to_numpy(dtype=float)

        if not np.isfinite(signal).all():
            continue

        features = calculate_features(signal)
        features["start_time_ms"] = window["time_ms"].iloc[0]
        features["end_time_ms"] = window["time_ms"].iloc[-1]
        results.append(features)

    if not results:
        print("Not enough valid samples for a complete feature window.")
        return None

    output = pd.DataFrame(results)
    output.to_csv(FEATURE_FILE, index=False)

    print(f"\nFeature windows created: {len(output)}")
    print(f"Feature file saved: {FEATURE_FILE}")
    print("\nLatest feature window:")
    print(output.tail(1).to_string(index=False))

    return output


def main():
    recording = collect_data()

    if recording is not None:
        extract_features(recording)


if __name__ == "__main__":
    main()
