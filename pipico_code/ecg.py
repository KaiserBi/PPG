from machine import ADC, Pin
import utime
import _thread
from time import sleep

upper_threshold = 560
lower_threshold = 530

last_beat_time = 0
waiting_for_beat = False
beat_detected = False
bpm = 0
prev_reading = 0
curr_reading = 1

adc = ADC(27)
lo_plus = Pin(17, Pin.IN)
lo_minus = Pin(16, Pin.IN)
filename = "ecg_data.csv"
with open(filename, "w") as f:
    f.write("time_ms,reading,bpm,leads_off\n")

print("Recording... press Ctrl+C to stop.")

def bpm_thread():
    global beat_detected, waiting_for_beat, bpm, last_beat_time, prev_reading, curr_reading
    while True:
        current_time = utime.ticks_ms()
        reading = round(adc.read_u16() / 64)

        if reading > upper_threshold:
            if beat_detected == True:
                bpm = round(1000 * (60 / (current_time - last_beat_time)))
                waiting_for_beat = False
                beat_detected = 0
            if waiting_for_beat == 0:
                last_beat_time = utime.ticks_ms()
                waiting_for_beat = True

        if reading < lower_threshold and waiting_for_beat:
            beat_detected = True

        prev_reading = curr_reading
        curr_reading = reading

        if curr_reading == prev_reading and bpm > 0:
            bpm = 0

        sleep(0.1)

_thread.start_new_thread(bpm_thread, ())

last_printed_bpm = -1

while True:
    try:
        timestamp = utime.ticks_ms()

        if lo_plus.value() == 1 or lo_minus.value() == 1:
            print("! Leads off detected")
            with open(filename, "a") as f:
                f.write("{},!,{},1\n".format(timestamp, bpm))
        else:
            reading = round(adc.read_u16() / 64)
            print(reading)

            with open(filename, "a") as f:
                f.write("{},{},{},0\n".format(timestamp, reading, bpm))

            if bpm != last_printed_bpm:
                print("BPM:", bpm)
                last_printed_bpm = bpm

        sleep(0.001)

    except KeyboardInterrupt:
        print("Done. Saved to", filename)
        break