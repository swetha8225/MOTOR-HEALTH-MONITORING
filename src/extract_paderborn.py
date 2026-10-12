from pathlib import Path
from scipy.io import loadmat
import numpy as np

DATASET_DIR = Path("data/paderborn")


def get_signal(file):

    mat = loadmat(
        file,
        squeeze_me=True,
        struct_as_record=False
    )

    variable_name = [
        key for key in mat.keys()
        if not key.startswith("__")
    ][0]

    data = mat[variable_name]

    # Paderborn X contains the recorded signals
    x_data = data.X

    print("\nFile:", file)
    print("Number of X signals:", len(x_data))

    for i, signal in enumerate(x_data):

        print(f"\nSignal {i}")

        if hasattr(signal, "_fieldnames"):

            print("Fields:", signal._fieldnames)

            for field in signal._fieldnames:

                value = getattr(signal, field)

                if hasattr(value, "shape"):
                    print(
                        f"  {field}: "
                        f"shape={value.shape}"
                    )
                else:
                    print(
                        f"  {field}: {value}"
                    )

    return data


def main():

    files = list(DATASET_DIR.rglob("*.mat"))

    if not files:
        print("No Paderborn MAT files found.")
        return

    # Use first real Paderborn file
    file = files[0]

    data = get_signal(file)

    print("\n" + "=" * 60)
    print("PADERBORN SIGNAL EXTRACTION TEST")
    print("=" * 60)

    # First X entry
    signal = data.X[0]

    print("\nFirst X entry:")

    if hasattr(signal, "_fieldnames"):

        for field in signal._fieldnames:

            value = getattr(signal, field)

            if isinstance(value, np.ndarray):

                print(f"{field}:")
                print("  shape:", value.shape)
                print("  samples:", len(value))

                if value.size > 0:
                    print("  first 10 values:")
                    print(value[:10])

    print("\nExtraction test complete.")


if __name__ == "__main__":
    main()
