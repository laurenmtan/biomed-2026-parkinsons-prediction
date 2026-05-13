import serial
import serial.tools.list_ports
import csv
import time
from datetime import datetime

# =============================================================================
# CONFIGURATION - Edit these settings
# =============================================================================
PORT = None  # Set to your port (e.g., '/dev/cu.usbmodem14101' or 'COM3'), or None to auto-detect
BAUD_RATE = 9600
NUM_INPUT_PORTS = 6  # Number of ports from Arduino

# Port to Sensor mapping: port_index -> sensor_number (1-indexed)
PORT_TO_SENSOR = {
    0: 14,
    1: 13,
    2: 8,
    3: 9,
    4: 16,
    5: 11,
}
# =============================================================================


def list_ports():
    """List and return available serial ports."""
    ports = serial.tools.list_ports.comports()
    print("Available ports:")
    for p in ports:
        print(f"  {p.device} - {p.description}")
    return [p.device for p in ports]


def map_to_16_sensors(raw_values):
    """
    Map 6 raw port values to 16 sensor columns.
    Port mapping: 0->14, 1->13, 2->8, 3->9, 4->16, 5->11
    Calculated: sensor_12 = avg(13,11), sensor_10 = avg(11,9)
    All others = 0
    """
    sensors = [0.0] * 16  # Initialize all 16 sensors to 0

    # Map raw port values to sensor positions
    for port_idx, sensor_num in PORT_TO_SENSOR.items():
        if port_idx < len(raw_values):
            sensors[sensor_num - 1] = raw_values[port_idx]  # Convert to 0-indexed

    # Calculate averaged sensors
    # Sensor 12 = avg(Sensor 13, Sensor 11)
    sensors[11] = (sensors[12] + sensors[10]) / 2  # sensor_12 = avg(sensor_13, sensor_11)
    # Sensor 10 = avg(Sensor 11, Sensor 9)
    sensors[9] = (sensors[10] + sensors[8]) / 2   # sensor_10 = avg(sensor_11, sensor_9)

    return sensors


def read_and_record():
    """Connect to Arduino and record sensor data to timestamped CSV."""

    # Find port
    port = PORT
    if port is None:
        ports = list_ports()
        if not ports:
            print("No serial ports found. Connect Arduino and try again.")
            return
        port = input("Enter port: ").strip()

    # Connect
    print(f"Connecting to {port}...")
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print("Connected!\n")

    # Create output file with timestamp
    filename = f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    # Create header with time first, then 16 sensors
    sensor_names = [f'sensor_{i}' for i in range(1, 17)]
    header = ['time'] + sensor_names

    print(f"Recording to: {filename}")
    print("Press Ctrl+C to stop\n")

    start_time = time.time()
    sample_count = 0

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()

                    # Parse comma-separated sensor values
                    try:
                        raw_values = [float(v) for v in line.split(',')]
                        if len(raw_values) >= NUM_INPUT_PORTS:
                            raw_values = raw_values[:NUM_INPUT_PORTS]

                            # Map to 16 sensors
                            sensors = map_to_16_sensors(raw_values)

                            # Elapsed time since recording started (in seconds)
                            elapsed = round(time.time() - start_time, 4)

                            writer.writerow([elapsed] + sensors)
                            f.flush()

                            sample_count += 1
                            if sample_count % 50 == 0:
                                print(f"[{elapsed:.1f}s] Sample {sample_count}")
                    except ValueError:
                        pass  # Skip malformed lines

        except KeyboardInterrupt:
            pass

    ser.close()

    duration = time.time() - start_time
    print(f"\n--- Recording Complete ---")
    print(f"Samples: {sample_count}")
    print(f"Duration: {duration:.1f}s")
    print(f"Rate: {sample_count/duration:.1f} Hz")
    print(f"Saved to: {filename}")


if __name__ == '__main__':
    read_and_record()