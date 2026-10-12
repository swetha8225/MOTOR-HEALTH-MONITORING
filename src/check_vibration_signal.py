
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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

for i, entry in enumerate(np.atleast_1d(data.Y)):
    name = getattr(entry, "Name", f"Y[{i}]")
    values = np.asarray(entry.Data).squeeze()

    print(f"\n{name}")
    print("Shape:", values.shape)
    print("Min:", np.min(values))
    print("Max:", np.max(values))
    print("Mean:", np.mean(values))
    print("Std:", np.std(values))

    if name == "vibration_1":
        plt.figure(figsize=(12, 4))
        plt.plot(values[:2000], linewidth=0.8)
        plt.title("Paderborn Bearing Vibration — vibration_1")
        plt.xlabel("Sample")
        plt.ylabel("Vibration amplitude")
        plt.grid(True)
        plt.tight_layout()

plt.show()
