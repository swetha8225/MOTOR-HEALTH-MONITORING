
import pandas as pd
from pathlib import Path
from datetime import datetime

INPUT = Path("results/motor_condition_comparison.csv")
OUTPUT = Path("results/motor_analysis_report.txt")


def main():
    df = pd.read_csv(INPUT, index_col=0)

    report = [
        "MOTOR HEALTH MONITORING - SENSOR ANALYSIS",
        "=" * 45,
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "Comparison of motor OFF and motor ON sensor features",
        "-" * 45,
    ]

    for feature, row in df.iterrows():
        off = row["Motor OFF"]
        on = row["Motor ON"]
        difference = row["Difference (ON - OFF)"]

        report.append(
            f"{feature}: OFF={off:.4f}, ON={on:.4f}, "
            f"difference={difference:+.4f}"
        )

    report.extend([
        "",
        "OBSERVATION",
        "-" * 45,
        "The motor-on measurements show substantially greater "
        "variation and signal range than the motor-off measurements.",
        "",
        "LIMITATION",
        "-" * 45,
        "These measurements indicate a difference between operating "
        "conditions, not a confirmed mechanical fault. Sensor mounting, "
        "orientation, electrical noise, and operating conditions may "
        "affect the readings. Further controlled testing is required.",
    ])

    OUTPUT.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print(f"\nReport saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
