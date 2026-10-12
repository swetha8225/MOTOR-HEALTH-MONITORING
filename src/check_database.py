
from database import get_connection, get_readings

with get_connection() as conn:
    count = conn.execute(
        "SELECT COUNT(*) FROM sensor_readings"
    ).fetchone()[0]

print("Total database readings:", count)
print("\nLatest 5 readings:")

for reading in get_readings(5):
    print(reading)
