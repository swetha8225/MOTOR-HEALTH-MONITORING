
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_FILE = Path("results/paderborn_labeled_features.csv")
MODEL_DIR = Path("results/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + ["condition", "bearing_id"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found. Validate the dataset first.")

    if not set(df["condition"].unique()).issubset(set(LABELS)):
        raise ValueError("Unexpected condition labels found.")

    # Every bearing must have exactly one assigned class.
    if (df.groupby("bearing_id")["condition"].nunique() != 1).any():
        raise ValueError("A bearing has multiple condition labels.")

    # Create one fixed split shared by every candidate model.
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

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    train_bearings = set(train["bearing_id"])
    test_bearings = set(test["bearing_id"])

    if train_bearings & test_bearings:
        raise RuntimeError("Bearing overlap detected!")

    for part_name, part in [("training", train), ("testing", test)]:
        absent = set(LABELS) - set(part["condition"].unique())
        if absent:
            raise ValueError(
                f"The {part_name} split is missing classes: {sorted(absent)}"
            )

    X_train, y_train = train[FEATURES], train["condition"]
    X_test, y_test = test[FEATURES], test["condition"]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Histogram Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42
            )
        )
    }

    results = []
    best_model = None
    best_name = None
    best_macro_f1 = -1

    print("=" * 60)
    print("BEARING CONDITION MODEL COMPARISON")
    print("=" * 60)
    print("Total windows:", len(df))
    print("Training windows:", len(train))
    print("Testing windows:", len(test))
    print("Training bearings:", len(train_bearings))
    print("Testing bearings:", len(test_bearings))
    print("Shared bearings:", len(train_bearings & test_bearings))

    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())

    print("\nTesting class counts:")
    print(y_test.value_counts().to_string())

    for name, classifier in models.items():
        print(f"\nTraining {name}...")

        # Imputation is fitted only on the training data.
        if name == "Histogram Gradient Boosting":
            model = classifier
        else:
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                classifier
            )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model = model
            best_name = name

    # Compare candidates on the same held-out bearings.
    comparison = pd.DataFrame(results).sort_values(
        "macro_f1", ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Re-evaluate the selected candidate for its detailed report.
    predictions = best_model.predict(X_test)

    print(f"\nBest candidate by macro F1: {best_name}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0
    ))

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(LABELS)
    print(confusion_matrix(y_test, predictions, labels=LABELS))

    # Save separately from the existing baseline model.
    candidate_path = MODEL_DIR / "bearing_condition_model_candidate.joblib"
    joblib.dump(best_model, candidate_path)

    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nBest candidate saved to:", candidate_path.resolve())
    print("Comparison saved to:", comparison_path.resolve())
    print("\nTraining and comparison complete.")


if __name__ == "__main__":
    main()
