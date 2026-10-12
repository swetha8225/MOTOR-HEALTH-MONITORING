
from pathlib import Path
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

TEST_FILE = Path("results/independent_test_data.csv")
MODEL_FILE = Path("results/models/independent_bearing_model.joblib")

FEATURES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
]

LABELS = ["healthy", "artificial_fault", "real_fault"]


def main():
    test_df = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_FILE)

    X_test = test_df[FEATURES]
    y_test = test_df["condition"]
    predictions = model.predict(X_test)

    print("=" * 55)
    print("INDEPENDENT MODEL EVALUATION")
    print("=" * 55)

    print("Test windows:", len(test_df))
    print("Unseen bearings:", sorted(test_df["bearing_id"].unique()))
    print("Number of unseen bearings:", test_df["bearing_id"].nunique())

    print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Macro F1:",
        round(f1_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ), 4),
    )

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print("Label order:", LABELS)
    print(confusion_matrix(y_test, predictions, labels=LABELS))


if __name__ == "__main__":
    main()
