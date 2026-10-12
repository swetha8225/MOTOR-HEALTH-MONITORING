
from pathlib import Path
import pandas as pd

DATA_FILE = Path("results/paderborn_labeled_features.csv")

TEST_BEARINGS = [
    "K005", "K006",
    "KA06", "KI07",
    "KA16", "KI16"
]


def main():
    df = pd.read_csv(DATA_FILE)

    test_df = df[df["bearing_id"].isin(TEST_BEARINGS)].copy()
    train_df = df[~df["bearing_id"].isin(TEST_BEARINGS)].copy()

    if set(train_df["bearing_id"]) & set(test_df["bearing_id"]):
        raise ValueError("Data leakage detected: bearing IDs overlap.")

    if set(test_df["bearing_id"]) != set(TEST_BEARINGS):
        raise ValueError("One or more requested test bearings are missing.")

    Path("results").mkdir(exist_ok=True)

    train_df.to_csv("results/independent_train_data.csv", index=False)
    test_df.to_csv("results/independent_test_data.csv", index=False)

    print("=" * 45)
    print("INDEPENDENT TEST SPLIT")
    print("=" * 45)
    print("Training bearings:", train_df["bearing_id"].nunique())
    print("Testing bearings:", test_df["bearing_id"].nunique())
    print("Training windows:", len(train_df))
    print("Testing windows:", len(test_df))

    print("\nTest bearings by condition:")
    print(
        test_df.groupby("condition")["bearing_id"]
        .nunique()
        .to_string()
    )

    print("\nSaved:")
    print("results/independent_train_data.csv")
    print("results/independent_test_data.csv")


if __name__ == "__main__":
    main()
