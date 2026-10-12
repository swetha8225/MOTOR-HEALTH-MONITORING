
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from feature_extraction import calculate_features
from preprocessing import create_windows

folder = Path("data/paderborn")
file = sorted(folder.rglob("*.mat"))[0]

mat = loadmat(file, squeeze_me=True, struct_as_record=False)
name = next(k for k in mat if not k.startswith("__"))
data = mat[name]

vibration = np.asarray(data.Y[6].Data, dtype=np.float64)

windows = create_windows(
    vibration,
    window_size=4000,
    step_size=4000
)

print("File:", file.name)
print("Channel:", data.Y[6].Name)
print("Samples:", len(vibration))
print("Windows:", len(windows))

for i, window in enumerate(windows[:4]):
    features = calculate_features(window)
    print(f"\nWindow {i}:")
    for key, value in features.items():
        print(f"  {key}: {value:.6f}")
