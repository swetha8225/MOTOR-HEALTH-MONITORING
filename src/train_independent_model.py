
from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

TRAIN_FILE = Path("results/independent_train_data.csv")
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


def main():
    df = pd.read_csv(TRAIN_FILE)

    X = df[FEATURES]
    y = df["condition"]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        )),
    ])

    print("Training independent model...")
    print("Training windows:", len(df))
    print("Training bearings:", df["bearing_id"].nunique())

    model.fit(X, y)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print("\nTraining completed successfully.")
    print("Saved model:", MODEL_FILE)
    print("Classes learned:", list(model.classes_))
    print("\nNext: evaluate this model on the independent test set.")


if __name__ == "__main__":
    main()
