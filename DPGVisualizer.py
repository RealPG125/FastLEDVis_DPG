import dearpygui.dearpygui as dpg
import time
import pyaudio
import numpy as np
import time
import threading
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, lfilter

CHUNK = 1300
RATE = 48000

pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16, 
    channels=2, 
    rate=RATE, 
    input=True, 
    input_device_index=3,
    frames_per_buffer=CHUNK, 
    )

floor = 100
bars = 32 
dataSizePerBox = (CHUNK / (2 * bars))
influnceMultipler = (1/ (64))
startFreq = 80
endFreq = 20000
high = startFreq
base = (endFreq / startFreq) ** (1 / bars)
velocity = np.empty(bars, dtype = int)
rawHeight = np.full(bars, 1, dtype = int)
height = np.full(bars, 1, dtype = int)

running = True

stream.stop_stream()
if not stream.is_active():
    stream.start_stream()

def bandpass_filter(data, lowcut, highcut, fs, order = 1):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype = 'bandpass')
    y = lfilter(b, a, data)
    return y

def audio_process():
    global rawHeight
    global left
    while (running):
        try:
            data = np.abs(np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16))
            left = data[::2]
            for i in range(bars):
                low = round(startFreq * base ** (i), 2)
                high = round(startFreq * base ** (i + 1), 2)
                rawHeight[i] = np.max(bandpass_filter(left, low, high, RATE)) * 1/64
        except:
            pass

def bars_update():
    for i in range(bars):
        if (rawHeight[i] > height[i]):
            velocity[i] = (rawHeight[i] * 0.3)
        height[i] += velocity[i] * 0.45
        if (height[i] <= 1 or height[i] > 150):
            height[i] = 1
        else:
            velocity[i] -= 3.5

t = threading.Thread(target = audio_process)
t.daemon = True
t.start()

# DPG

def frame():
    bars_update()
    for i in range(bars):
       dpg.configure_item(f"height_text{i}", text = str(rawHeight[i]))
       dpg.configure_item(f"dynamic_box{i}", p1 = (10 + 15 * i,floor), p2 = (20 + 15 * i,floor), p3 = (20 + 15 * i,floor - height[i]), p4 = (10 + 15 * i,floor - height[i]))
    dpg.configure_item("window_size_text", text = f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15)

dpg.create_context()

def exit_program():
    print("exiting program...")
    stream.stop_stream()
    dpg.destroy_context()
    quit()

with dpg.window(label="Main", tag = "main_window", width = 515, height = 250):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i,floor + 10), str(rawHeight), color=(250, 250, 250, 255), size=15, tag = f"height_text{i}")
        dpg.draw_quad((10 + 15 * i,floor), (20 + 15 * i,floor), (20 + 15 * i,floor - height[i]), (10 + 15 * i,floor - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
    dpg.draw_text((10, floor + 25), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
    dpg.add_button(pos = (30, floor + 80), label = "exit", tag = "exit_button")

with dpg.item_handler_registry(tag = "exit_handler") as handler:
    dpg.add_item_clicked_handler(callback = exit_program)

dpg.bind_item_handler_registry("exit_button", "exit_handler")

dpg.set_viewport_vsync = True

dpg.create_viewport(title='DPGVisualizer', width=530, height=250)
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    frame()
    dpg.render_dearpygui_frame()
    time.sleep(1/180)