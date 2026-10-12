
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "results" / "motor_health.db"

DB_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_connection():
    return sqlite3.connect(DB_FILE)


def create_table():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_ms REAL NOT NULL,
                ax REAL NOT NULL,
                ay REAL NOT NULL,
                az REAL NOT NULL,
                total_accel REAL NOT NULL,
                acs_adc REAL NOT NULL
            )
        """)


def save_reading(reading):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sensor_readings
            (time_ms, ax, ay, az, total_accel, acs_adc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, reading)


def get_readings(limit=100):
    with get_connection() as conn:
        return conn.execute("""
            SELECT time_ms, ax, ay, az, total_accel, acs_adc
            FROM sensor_readings
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()


if __name__ == "__main__":
    create_table()
    print("Database created successfully!")
    print(f"Location: {DB_FILE}")
