
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = Path("results/paderborn_labeled_features.csv")
MODEL_FILE = Path("results/models/bearing_condition_model_candidate.joblib")

FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor"
]

LABELS = ["healthy", "artificial_fault", "real_fault"]


def main():
    df = pd.read_csv(DATA_FILE)

    splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            df[FEATURES],
            df["condition"],
            groups=df["bearing_id"]
        )
    )

    test = df.iloc[test_idx]

    model = joblib.load(MODEL_FILE)
    predictions = model.predict(test[FEATURES])
    actual = test["condition"]

    print("=" * 55)
    print("SAVED MODEL EVALUATION")
    print("=" * 55)
    print("Test windows:", len(test))
    print("Held-out bearings:", sorted(test["bearing_id"].unique()))
    print("Accuracy:", round(accuracy_score(actual, predictions), 4))
    print(
        "Macro F1:",
        round(
            f1_score(
                actual,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0
            ),
            4
        )
    )

    print("\nClassification report:")
    print(classification_report(
        actual,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(LABELS)
    print(confusion_matrix(actual, predictions, labels=LABELS))


if __name__ == "__main__":
    main()
