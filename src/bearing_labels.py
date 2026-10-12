
from pathlib import Path
import pandas as pd

# Healthy reference bearings

HEALTHY = {
    "K001", "K002", "K003",
    "K004", "K005", "K006"
}

ARTIFICIAL_FAULT = {
    "KA01", "KA03", "KA05", "KA06",
    "KA07", "KA08", "KA09",
    "KI01", "KI03", "KI05", "KI07", "KI08"
}

REAL_FAULT = {
    "KA04", "KA15", "KA16", "KA22", "KA30",
    "KB23", "KB24", "KB27",
    "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"
}

LABEL_MAP = {}

for bearing in HEALTHY:
    LABEL_MAP[bearing] = "healthy"

for bearing in ARTIFICIAL_FAULT:
    LABEL_MAP[bearing] = "artificial_fault"

for bearing in REAL_FAULT:
    LABEL_MAP[bearing] = "real_fault"


def main():
    data_dir = Path("data/paderborn")
    folders = sorted(
        folder.name
        for folder in data_dir.iterdir()
        if folder.is_dir() and list(folder.glob("*.mat"))
    )

    print("=" * 55)
    print("BEARING LABEL MAPPING CHECK")
    print("=" * 55)

    for folder in folders:
        label = LABEL_MAP.get(folder, "UNMAPPED")
        print(f"{folder:8s} -> {label}")

    print("\nTotal folders:", len(folders))
    print("Mapped folders:", sum(f in LABEL_MAP for f in folders))
    print("Unmapped folders:", sum(f not in LABEL_MAP for f in folders))

    print("\nLabel counts:")
    for label in ("healthy", "artificial_fault", "real_fault"):
        print(f"{label:20s}: {sum(LABEL_MAP.get(f) == label for f in folders)}")


if __name__ == "__main__":
    main()
