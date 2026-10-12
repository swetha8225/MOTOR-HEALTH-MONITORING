
import csv
import time
import serial
from pathlib import Path

from database import create_table, save_reading

PORT = "COM8"
BAUD_RATE = 115200

ROOT = Path(__file__).resolve().parent.parent
HEADERS = ["time_ms", "ax", "ay", "az", "total_accel", "acs_adc"]


def main():
    create_table()

    mode = input("Recording mode (off/on): ").strip().lower()

    if mode not in ("off", "on"):
        print("Enter either off or on.")
        return

    csv_file = ROOT / "results" / f"motor_{mode}.csv"
    duration = 30

    print(f"Recording {mode.upper()} data for {duration} seconds.")
    print("Close Arduino Serial Monitor before continuing.")

    try:
        with serial.Serial(PORT, BAUD_RATE, timeout=2) as esp:
            print("Connected!")
            time.sleep(2)
            esp.reset_input_buffer()

            csv_file.parent.mkdir(parents=True, exist_ok=True)

            with csv_file.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(HEADERS)
                file.flush()

                start = time.time()
                count = 0

                while time.time() - start < duration:
                    line = esp.readline().decode(
                        "utf-8", errors="replace"
                    ).strip()

                    if not line or line.startswith("#") or line.startswith("time_ms"):
                        continue

                    parts = line.split(",")
                    if len(parts) != len(HEADERS):
                        continue

                    try:
                        reading = [float(value) for value in parts]
                    except ValueError:
                        continue

                    writer.writerow(reading)
                    file.flush()
                    save_reading(reading)
                    count += 1

            print(f"\nSaved {count} readings to {csv_file}")

    except serial.SerialException as error:
        print("Serial connection error:", error)

    except KeyboardInterrupt:
        print("\nRecording interrupted.")

    print("Done.")


if __name__ == "__main__":
    main()
