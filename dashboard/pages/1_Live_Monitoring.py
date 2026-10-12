
from pathlib import Path
import csv
import io
import time

import pandas as pd
import streamlit as st

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RECORDING_PATH = RESULTS / "latest_motor_recording.csv"

COLUMNS = [
    "time_ms",
    "ax",
    "ay",
    "az",
    "total_accel",
    "acs_adc",
]

st.set_page_config(
    page_title="Live Motor Monitoring",
    page_icon="📡",
    layout="wide",
)

st.title("📡 ESP32 Live Motor Monitoring")
st.caption("Direct serial capture • MPU6050 • ACS712 • 30-second recording")

st.warning(
    "Experimental project: this page records real ESP32 readings. "
    "The bearing model has not been validated for reliable blender "
    "fault diagnosis."
)

if serial is None:
    st.error(
        "PySerial is not installed in this environment. Install it using "
        "the command shown below, then restart the dashboard."
    )
    st.code(".\\.venv\\Scripts\\python.exe -m pip install pyserial")
    st.stop()

# --------------------------------------------------
# Serial port configuration
# --------------------------------------------------
st.subheader("1. Connect to ESP32")

available_ports = []
if list_ports is not None:
    available_ports = [
        port.device for port in list_ports.comports()
    ]

default_port = "COM8" if "COM8" in available_ports else (
    available_ports[0] if available_ports else "COM8"
)

col1, col2, col3 = st.columns(3)

with col1:
    port = st.selectbox(
        "ESP32 serial port",
        options=available_ports if available_ports else [default_port],
        index=(
            available_ports.index(default_port)
            if default_port in available_ports
            else 0
        ),
        help="Choose the COM port used by your ESP32.",
    )

with col2:
    baud_rate = st.selectbox(
        "Baud rate",
        options=[115200, 9600, 57600, 230400],
        index=0,
    )

with col3:
    duration = st.selectbox(
        "Capture duration",
        options=[30],
        format_func=lambda x: f"{x} seconds",
    )

if available_ports:
    st.caption("Detected serial ports: " + ", ".join(available_ports))
else:
    st.info(
        "No serial ports were detected. Connect the ESP32 and check "
        "Windows Device Manager. You can refresh this page to scan again."
    )

st.info(
    "Close Arduino Serial Monitor and any other program using this COM port "
    "before capturing."
)

# --------------------------------------------------
# Live display placeholders
# --------------------------------------------------
status_area = st.empty()
progress_area = st.empty()
metrics_area = st.empty()
chart_area = st.empty()
table_area = st.empty()

# --------------------------------------------------
# Capture helper
# --------------------------------------------------


def parse_sensor_line(line):
    """Parse one firmware CSV line; ignore headers and invalid messages."""
    try:
        parts = next(csv.reader([line.strip()]))
        if len(parts) < 6:
            return None

        values = [float(value.strip()) for value in parts[:6]]

        if not all(pd.notna(value) for value in values):
            return None

        return dict(zip(COLUMNS, values))

    except (ValueError, csv.Error, StopIteration):
        return None


def capture_serial(port_name, baud, seconds):
    """Capture serial CSV rows for a fixed time interval."""
    rows = []
    start = time.monotonic()
    deadline = start + seconds
    last_ui_update = 0.0

    with serial.Serial(
        port=port_name,
        baudrate=baud,
        timeout=0.2,
    ) as connection:
        # Allow the ESP32 serial connection to settle after opening.
        time.sleep(1.0)
        connection.reset_input_buffer()

        while time.monotonic() < deadline:
            raw = connection.readline()

            if raw:
                line = raw.decode("utf-8", errors="ignore").strip()
                row = parse_sensor_line(line)

                if row is not None:
                    rows.append(row)

            now = time.monotonic()
            elapsed = min(seconds, now - start)

            if now - last_ui_update >= 0.5:
                progress_area.progress(
                    min(1.0, elapsed / seconds),
                    text=(
                        f"Capturing: {elapsed:.1f} / {seconds} seconds"
                        f" • Valid samples: {len(rows)}"
                    ),
                )

                if rows:
                    current = pd.DataFrame(rows)
                    latest = current.iloc[-1]

                    with metrics_area.container():
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Valid samples", len(rows))
                        m2.metric("Acceleration X", f"{latest['ax']:.4f}")
                        m3.metric("Acceleration Y", f"{latest['ay']:.4f}")
                        m4.metric("Acceleration Z", f"{latest['az']:.4f}")

                    with chart_area.container():
                        st.subheader("Incoming vibration readings")
                        st.line_chart(
                            current.tail(300).set_index("time_ms")[
                                ["ax", "ay", "az", "total_accel"]
                            ]
                        )

                        st.subheader("Incoming ACS712 readings")
                        st.line_chart(
                            current.tail(300).set_index("time_ms")[["acs_adc"]]
                        )

                    with table_area.container():
                        st.subheader("Latest incoming samples")
                        st.dataframe(
                            current.tail(10),
                            use_container_width=True,
                            hide_index=True,
                        )

                last_ui_update = now

    return pd.DataFrame(rows, columns=COLUMNS)


# --------------------------------------------------
# Start capture
# --------------------------------------------------
st.subheader("2. Live capture")

if st.button("▶ Start 30-second live capture", type="primary"):
    RESULTS.mkdir(parents=True, exist_ok=True)

    try:
        status_area.info(
            f"Opening {port} at {baud_rate} baud. "
            "Keep the ESP32 connected for the full capture."
        )

        captured = capture_serial(port, baud_rate, duration)

        if captured.empty:
            status_area.error(
                "No valid sensor rows were received. Check the COM port, "
                "baud rate, firmware output format, and USB connection."
            )
        else:
            captured.to_csv(RECORDING_PATH, index=False)

            status_area.success(
                f"Capture complete: {len(captured)} valid samples saved to "
                f"{RECORDING_PATH.relative_to(ROOT)}"
            )

            st.subheader("3. Capture summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Valid samples", f"{len(captured):,}")
            c2.metric(
                "First ESP32 timestamp",
                f"{captured['time_ms'].iloc[0]:.0f} ms",
            )
            c3.metric(
                "Last ESP32 timestamp",
                f"{captured['time_ms'].iloc[-1]:.0f} ms",
            )

            st.dataframe(
                captured.tail(20),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download captured recording",
                data=captured.to_csv(index=False).encode("utf-8"),
                file_name="latest_motor_recording.csv",
                mime="text/csv",
            )

            st.info(
                "The recording is saved. Run your existing feature-processing "
                "workflow next so latest_motor_features.csv is generated from "
                "this new recording before running a model prediction."
            )

    except serial.SerialException as exc:
        status_area.error(
            f"Could not access {port}: {exc}. Close Arduino Serial Monitor, "
            "check the USB connection, and confirm the selected COM port."
        )
    except Exception as exc:
        status_area.error(f"Capture failed: {exc}")


# --------------------------------------------------
# Previously saved recording
# --------------------------------------------------
st.divider()
st.subheader("Previously saved recording")

if RECORDING_PATH.exists():
    try:
        saved = pd.read_csv(RECORDING_PATH)

        st.write(f"Saved samples: {len(saved):,}")
        st.dataframe(
            saved.tail(10),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download saved recording",
            data=saved.to_csv(index=False).encode("utf-8"),
            file_name="latest_motor_recording.csv",
            mime="text/csv",
            key="download_saved_recording",
        )
    except Exception as exc:
        st.error(f"Could not read the saved recording: {exc}")
else:
    st.caption("No saved recording is available yet.")
