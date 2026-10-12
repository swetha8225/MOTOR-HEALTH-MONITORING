
from pathlib import Path
from scipy.io import loadmat

dataset_dir = Path("data/paderborn")

bearing_ids = ["K001", "K006", "KA01", "KA08", "KB23", "KI01", "KI21"]


def print_fields(obj, indent=0):
    """Print the actual contents of a MATLAB metadata structure."""
    prefix = " " * indent

    if hasattr(obj, "_fieldnames"):
        for field in obj._fieldnames:
            value = getattr(obj, field)
            print(f"{prefix}{field}:")

            if hasattr(value, "_fieldnames"):
                print_fields(value, indent + 4)
            else:
                print(f"{prefix}    {value}")

    else:
        print(f"{prefix}{obj}")


print("=" * 60)
print("PADERBORN BEARING METADATA INSPECTION")
print("=" * 60)

for bearing_id in bearing_ids:
    files = sorted((dataset_dir / bearing_id).glob("*.mat"))

    if not files:
        print(f"\n{bearing_id}: No files found")
        continue

    file = files[0]
    print(f"\n{'-' * 50}")
    print("Bearing folder:", bearing_id)
    print("File:", file.name)

    try:
        mat = loadmat(
            file,
            squeeze_me=True,
            struct_as_record=False
        )

        variable_name = next(
            key for key in mat if not key.startswith("__")
        )
        data = mat[variable_name]

        print("\nDescription:")
        print_fields(data.Description)

    except Exception as error:
        print("Could not inspect this file:", error)

print("\nMetadata inspection complete.")
