
from pathlib import Path
from collections import Counter

dataset_dir = Path("data/paderborn")

mat_files = list(dataset_dir.rglob("*.mat"))

folder_counts = Counter(
    file.parent.name for file in mat_files
)

print("=" * 60)
print("PADERBORN BEARING FOLDER SUMMARY")
print("=" * 60)

print("Total MAT files:", len(mat_files))
print("Unique bearing folders:", len(folder_counts))

print("\nFiles per folder:\n")

for folder, count in sorted(folder_counts.items()):
    print(f"{folder:10s} {count:5d}")

print("\nFolder inspection complete.")
