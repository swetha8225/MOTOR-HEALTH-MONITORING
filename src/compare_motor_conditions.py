
import pandas as pd
from pathlib import Path

FILES = {
    "Motor OFF": Path("results/motor_off_features.csv"),
    "Motor ON": Path("results/motor_on_features.csv"),
}

FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
]


def main():
    summaries = {}

    for condition, path in FILES.items():
        df = pd.read_csv(path)
        summaries[condition] = df[FEATURES].mean()

    comparison = pd.DataFrame(summaries)
    comparison["Difference (ON - OFF)"] = (
        comparison["Motor ON"] - comparison["Motor OFF"]
    )

    output = Path("results/motor_condition_comparison.csv")
    comparison.to_csv(output)

    print("\nMOTOR OFF vs MOTOR ON")
    print(comparison.round(4).to_string())
    print(f"\nSaved comparison to: {output}")


if __name__ == "__main__":
    main()
