
from pathlib import Path
from scipy.io import loadmat
from preprocessing import create_windows

folder = Path("data/paderborn")
files = sorted(folder.rglob("*.mat"))

if not files:
    raise FileNotFoundError("No Paderborn MAT files found.")

file = files[0]
mat = loadmat(file, squeeze_me=True, struct_as_record=False)

name = next(k for k in mat if not k.startswith("__"))
signal = mat[name].X[0].Data

windows = create_windows(signal, window_size=4000, step_size=4000)

print("File:", file.name)
print("Total signal samples:", len(signal))
print("Window size:", 4000)
print("Number of complete windows:", len(windows))

if windows:
    print("First window samples:", len(windows[0]))
