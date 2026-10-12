from src.motor_feature_utils import calculate_features
from pathlib import Path
import pandas as pd


WINDOW_SIZE = 100
AXIS = "az"


def process_motor_csv(input_file, output_file=None):
    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(f"File not found: {input_file}")

    df = pd.read_csv(input_file)

    required = {"time_ms", "ax", "ay", "az"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    results = []

    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        signal = window[AXIS].to_numpy(dtype=float)

        if not window["time_ms"].notna().all():
            continue

        if not pd.Series(signal).map(
            lambda x: pd.notna(x) and abs(x) != float("inf")
        ).all():
            continue

        features = calculate_features(signal)
        features["start_time_ms"] = window["time_ms"].iloc[0]
        features["end_time_ms"] = window["time_ms"].iloc[-1]
        results.append(features)

    if not results:
        raise ValueError("No valid feature windows could be created.")

    output = pd.DataFrame(results)

    if output_file is None:
        output_file = input_file.with_name(
            input_file.stem + "_features.csv"
        )

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_file, index=False)

    return output


if __name__ == "__main__":
    for filename in ["results/motor_off.csv", "results/motor_on.csv"]:
        output = process_motor_csv(filename)
        print(f"{filename}: {len(output)} windows processed.")
