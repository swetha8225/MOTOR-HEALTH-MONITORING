
from pathlib import Path
import joblib
import pandas as pd

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
    if not MODEL_FILE.exists():
        print("Model file not found:", MODEL_FILE)
        return

    model = joblib.load(MODEL_FILE)

    print("=" * 45)
    print("BEARING CONDITION PREDICTION")
    print("=" * 45)
    print("Enter the seven features from one vibration window.")

    values = {}

    try:
        for feature in FEATURES:
            values[feature] = float(input(f"{feature}: "))

        sample = pd.DataFrame([values], columns=FEATURES)
        prediction = model.predict(sample)[0]

        print("\nPredicted condition:", prediction)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            print("\nModel scores:")
            for label, probability in zip(model.classes_, probabilities):
                print(f"{label}: {probability:.2%}")

        print("\nNote: This is a Paderborn-trained model.")
        print("Do not treat this prediction as a diagnosis of your blender.")

    except ValueError:
        print("Invalid input. Please enter numbers only.")


if __name__ == "__main__":
    main()
