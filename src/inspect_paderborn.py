
from pathlib import Path
from scipy.io import loadmat

DATASET_DIR = Path("data/paderborn")


def print_fields(obj, indent="  "):
    """Print the fields of a MATLAB structure."""
    if not hasattr(obj, "_fieldnames"):
        print(f"{indent}{obj}")
        return

    for field in obj._fieldnames:
        value = getattr(obj, field)

        if hasattr(value, "_fieldnames"):
            print(f"{indent}{field}:")
            print_fields(value, indent + "  ")

        elif hasattr(value, "shape"):
            print(f"{indent}{field}: shape={value.shape}")

        else:
            print(f"{indent}{field}: {value}")


def main():
    print("=" * 70)
    print("PADERBORN SIGNAL INSPECTION")
    print("=" * 70)

    # Find MAT files
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR.resolve()}"
        )

    files = sorted(DATASET_DIR.rglob("*.mat"))

    if not files:
        raise FileNotFoundError(
            f"No MAT files found in {DATASET_DIR.resolve()}"
        )

    file = files[0]
    print("\nFile:")
    print(file)

    # Load MAT file
    mat = loadmat(
        file,
        squeeze_me=True,
        struct_as_record=False
    )

    # Find dataset variable
    variable_names = [
        key for key in mat
        if not key.startswith("__")
    ]

    if not variable_names:
        raise ValueError("No dataset variable found in the MAT file.")

    variable_name = variable_names[0]
    data = mat[variable_name]

    print("\nDataset variable:")
    print(variable_name)

    # Inspect main fields
    print("\nMAIN DATASET FIELDS")
    print("=" * 50)

    for field in data._fieldnames:
        value = getattr(data, field)

        print(f"\nField: {field}")
        print(f"Type: {type(value)}")

        if hasattr(value, "shape"):
            print(f"Shape: {value.shape}")

        if field in ["Info", "Description"]:
            print_fields(value)

        elif field in ["X", "Y"]:
            print(f"Number of entries: {len(value)}")

            for i, entry in enumerate(value):
                print(f"\n  Entry {i}:")
                print(f"  Type: {type(entry)}")

                if hasattr(entry, "_fieldnames"):
                    print(f"  Fields: {entry._fieldnames}")
                    print_fields(entry, "    ")

    # Inspect general metadata
    print("\nINFO METADATA")
    print("=" * 50)
    print_fields(data.Info)

    # Inspect vibration channel
    print("\nVIBRATION CHANNEL METADATA")
    print("=" * 50)

    if len(data.Y) > 6:
        vibration = data.Y[6]

        for field in vibration._fieldnames:
            value = getattr(vibration, field)

            if field == "Data":
                print(f"{field}: shape={value.shape}")
            else:
                print(f"{field}: {value}")
    else:
        print("Y[6] is not available in this file.")

    # Inspect measurement description
    print("\nMEASUREMENT DESCRIPTION")
    print("=" * 50)
    print_fields(data.Description)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
