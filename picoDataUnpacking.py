import serial
import struct
import csv

# Open serial port
ser = serial.Serial('COM8', 115200, timeout=1)

# File to save data
with open("ppg_data.csv", "w", newline='') as f:
    writer = csv.writer(f)

    print("Recording...")
    buffer = b''

    try:
        while True:
            buffer += ser.read(64)  # Read up to 64 bytes at a time

            # Look for complete packets (start with 0xAA, 6 bytes total)
            while len(buffer) >= 7:
                if buffer[0] != 0xAA:
                    buffer = buffer[1:]
                    continue

                packet = buffer[:7]
                if len(packet) < 7:
                    break

                timestamp_us, sample = struct.unpack('<IH', packet[1:])  # skip sync byte
                writer.writerow([timestamp_us, sample])
                print(f"t={timestamp_us} µs, sample={sample}")

                buffer = buffer[7:]

    except KeyboardInterrupt:
        print("Stopped.")
        ser.close()