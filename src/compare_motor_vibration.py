import pandas as pd
import numpy as np

FILE = "results/esp32_vibration.csv"

df = pd.read_csv(FILE)

print("Choose which recording to analyze:")
print("1. Motor OFF")
print("2. Motor ON")

choice = input("Enter 1 or 2: ").strip()

start = input("Starting row number (0-based): ").strip()
end = input("Ending row number (exclusive): ").strip()

try:
    start, end = int(start), int(end)
    if choice not in ("1", "2") or start < 0 or end <= start:
        raise ValueError

    data = df.iloc[start:end]["ax"].dropna().to_numpy()
    if len(data) < 10:
        print("Not enough samples. Select at least 10 rows.")
        raise SystemExit

    vibration = data - np.mean(data)
    rms = np.sqrt(np.mean(vibration ** 2))

    print("\nRecording:", "Motor OFF" if choice == "1" else "Motor ON")
    print("Samples:", len(data))
    print(f"Vibration RMS: {rms:.6f} m/s²")
    print(f"Peak-to-peak: {np.ptp(vibration):.6f} m/s²")
    print(f"Peak: {np.max(np.abs(vibration)):.6f} m/s²")

except ValueError:
    print("Invalid input. Enter a valid choice and row range.")
