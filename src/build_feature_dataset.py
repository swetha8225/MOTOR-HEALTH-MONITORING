
from preprocessing import create_windows
from feature_extraction import calculate_features
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

# Allow imports from the src folder
sys.path.insert(0, str(Path(__file__).resolve().parent))


DATASET_DIR = Path("data/paderborn")
OUTPUT_DIR = Path("results")
WINDOW_SIZE = 4000
STEP_SIZE = 4000


def main():
    files = sorted(DATASET_DIR.rglob("*.mat"))

    if not files:
        raise FileNotFoundError(
            f"No MAT files found under {DATASET_DIR.resolve()}"
        )

    rows = []
    failed_files = []

    print(f"Found {len(files)} MAT files.")
    print("Processing vibration recordings...")

    for index, file in enumerate(files, start=1):
        try:
            mat = loadmat(
                file,
                squeeze_me=True,
                struct_as_record=False
            )

            variable_names = [
                key for key in mat
                if not key.startswith("__")
            ]

            if not variable_names:
                failed_files.append((str(file), "No dataset variable"))
                continue

            data = mat[variable_names[0]]

            # Use the first mechanical signal.
            vibration = np.asarray(
                data.Y[6].Data,
                dtype=np.float64
            )

            windows = create_windows(
                vibration,
                window_size=WINDOW_SIZE,
                step_size=STEP_SIZE
            )

            for window_number, window in enumerate(windows):
                features = calculate_features(window)

                rows.append({
                    "file": file.name,
                    "bearing_id": file.parent.name,
                    "window": window_number,
                    **features
                })

        except Exception as error:
            failed_files.append((str(file), str(error)))

        if index % 100 == 0:
            print(f"Processed {index}/{len(files)} files...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "paderborn_features.csv"

    if not rows:
        raise RuntimeError(
            "No feature rows were created. Check the MAT structure."
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print("\nFeature dataset created.")
    print("Output:", output_file.resolve())
    print("Feature rows:", len(df))
    print("Unique source files:", df["file"].nunique())
    print("Feature columns:", list(df.columns))

    if failed_files:
        print("Files with errors:", len(failed_files))
        print("First few errors:")
        for filename, error in failed_files[:5]:
            print(filename, "->", error)

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
