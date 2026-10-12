
import pandas as pd
import numpy as np

for name in ["motor_off", "motor_on"]:
    df = pd.read_csv(f"results/{name}.csv")

    print(f"\n{name.upper()}")
    print("-" * 40)

    for axis in ["ax", "ay", "az"]:
        x = df[axis].dropna().to_numpy()
        x = x - np.mean(x)

        rms = np.sqrt(np.mean(x ** 2))
        peak_to_peak = np.ptp(x)

        print(
            f"{axis}: RMS={rms:.4f} m/s², "
            f"peak-to-peak={peak_to_peak:.4f} m/s²"
        )
