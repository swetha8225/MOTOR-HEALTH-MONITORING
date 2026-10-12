
from pathlib import Path
import numpy as np
from scipy.io import loadmat

folder = Path("data/paderborn")
files = sorted(folder.rglob("*.mat"))

if not files:
    raise FileNotFoundError("No MAT files found.")

file = files[0]
mat = loadmat(file, squeeze_me=True, struct_as_record=False)

name = next(k for k in mat if not k.startswith("__"))
data = mat[name]

print("File:", file.name)
print("\nRecorded measurement channels (Y):")

for i, entry in enumerate(data.Y):
    print(f"\nY[{i}]")
    print("Name:", getattr(entry, "Name", "Unknown"))
    print("Raster:", getattr(entry, "Raster", "Unknown"))

    values = np.asarray(entry.Data).squeeze()

    print("Shape:", values.shape)

    if values.size and values.ndim == 1:
        print("First 10 values:", values[:10])
        print("Min:", np.min(values))
        print("Max:", np.max(values))
        print("Mean:", np.mean(values))
        print("Std:", np.std(values))
