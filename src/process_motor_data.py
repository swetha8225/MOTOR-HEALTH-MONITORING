
import pandas as pd
from pathlib import Path
from src.motor_feature_utils import calculate_features

WINDOW_SIZE = 100
INPUT_FILES = [
    Path("results/motor_off.csv"),
    Path("results/motor_on.csv"),
]


def process_csv(input_file):
    df = pd.read_csv(input_file)

    required = {"time_ms", "ax", "ay", "az"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{input_file}: missing columns {missing}")

    results = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]

        signal = window["az"].to_numpy()

        if not window["time_ms"].notna().all():
            continue
        if not pd.Series(signal).map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        ).all():
            continue

        features = calculate_features(signal)
        features["start_time_ms"] = window["time_ms"].iloc[0]
        features["end_time_ms"] = window["time_ms"].iloc[-1]
        results.append(features)

    if not results:
        raise ValueError(f"No valid windows in {input_file}")

    output_file = input_file.with_name(
        input_file.stem + "_features.csv"
    )
    output = pd.DataFrame(results)
    output.to_csv(output_file, index=False)

    print(f"\nFile: {input_file}")
    print(f"Samples: {len(df)}")
    print(f"Windows: {len(output)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    for file in INPUT_FILES:
        process_csv(file)
