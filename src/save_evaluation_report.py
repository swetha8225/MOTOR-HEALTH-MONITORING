
from pathlib import Path
from datetime import date
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
OUTPUT_FILE = Path("results/independent_evaluation_report.txt")

FEATURES = [
    "mean", "std", "rms", "peak",
    "peak_to_peak", "kurtosis", "crest_factor"
]
LABELS = ["healthy", "artificial_fault", "real_fault"]


def main():
    test_df = pd.read_csv(TEST_FILE)
    model = joblib.load(MODEL_FILE)

    actual = test_df["condition"]
    predicted = model.predict(test_df[FEATURES])

    lines = [
        "INDEPENDENT BEARING MODEL EVALUATION",
        "=" * 45,
        f"Report date: {date.today()}",
        f"Model: {MODEL_FILE}",
        f"Test dataset: {TEST_FILE}",
        f"Test windows: {len(test_df)}",
        f"Unseen bearings: {test_df['bearing_id'].nunique()}",
        f"Bearing IDs: {', '.join(sorted(test_df['bearing_id'].unique()))}",
        f"Accuracy: {accuracy_score(actual, predicted):.4f}",
        f"Macro F1: {f1_score(actual, predicted, labels=LABELS, average='macro', zero_division=0):.4f}",
        "",
        "CLASSIFICATION REPORT",
        classification_report(
            actual, predicted, labels=LABELS,
            target_names=LABELS, zero_division=0
        ),
        "CONFUSION MATRIX (rows=actual, columns=predicted)",
        f"Label order: {LABELS}",
        str(confusion_matrix(actual, predicted, labels=LABELS)),
        "",
        "LIMITATIONS",
        "- The test set contains six bearings.",
        "- Windows from one bearing are not independent observations.",
        "- Results depend on the dataset labels and selected bearings.",
        "- Blender-specific performance has not been established.",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("Report saved to:", OUTPUT_FILE.resolve())


if __name__ == "__main__":
    main()
