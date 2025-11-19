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
DEVICE_ID = 3

pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16, 
    channels=2, 
    rate=RATE, 
    input=True, 
    input_device_index=DEVICE_ID,
    frames_per_buffer=CHUNK, 
    )

fps = 180

baseYPos = 150
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

def bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype = 'bandpass')
    y = lfilter(b, a, data)
    return y

def audio_process():
    global rawHeight
    global left
    useOrder = 1

    while (running):
        try:
            useOrder = dpg.get_value(filterOrderSlider)
        except:
            pass
        try:
            data = np.abs(np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16))
            left = data[::2]
            for i in range(bars):
                low = round(startFreq * base ** (i), 2)
                high = round(startFreq * base ** (i + 1), 2)
                rawHeight[i] = np.max(bandpass_filter(left, low, high, RATE, useOrder)) * 1/64
        except:
            pass

def bars_update():
    decaySpeed = dpg.get_value(decaySpeedSlider)
    powerMultiplier = dpg.get_value(powerMultiplierSlider)
    if (decaySpeed == None or powerMultiplier == None):
        decaySpeed = 1
        powerMultiplier = 1
    for i in range(bars):
        if (rawHeight[i] > height[i]):
            velocity[i] = rawHeight[i]
        height[i] += velocity[i] * powerMultiplier * 0.15 * (180 / fps)
        if (height[i] <= 1 or height[i] > 1000):
            height[i] = 1
        else:
            velocity[i] -= decaySpeed * 5 * (180 / fps)

t = threading.Thread(target = audio_process)
t.daemon = True
t.start()

# DPG

def frame():
    fps = dpg.get_value(fpsSlider)
    bars_update()
    for i in range(bars):
       dpg.configure_item(f"height_text_raw{i}", text = str(rawHeight[i]))
       dpg.configure_item(f"height_text{i}", text = str(height[i]))
       dpg.configure_item(f"dynamic_box_raw{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - rawHeight[i]), p4 = (10 + 15 * i, baseYPos - rawHeight[i]))
       dpg.configure_item(f"dynamic_box{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - height[i]), p4 = (10 + 15 * i, baseYPos - height[i]))
    dpg.configure_item("window_size_text", text = f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15)

dpg.create_context()

def exit_program():
    print("exiting program...")
    stream.stop_stream()
    dpg.destroy_context()
    quit()

with dpg.window(label = "Main", tag = "main_window", width = 515, height = 370):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(height[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - height[i]), (10 + 15 * i, baseYPos - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
    dpg.draw_text((10, baseYPos + 30), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
    dpg.add_button(pos = (30, baseYPos + 80), label = "exit", tag = "exit_button")
    decaySpeedSlider = dpg.add_slider_float(label = "decay speed", pos = (30, baseYPos + 120), default_value = 1, min_value = 0.25, max_value = 4)
    powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", pos = (30, baseYPos + 140), default_value = 1, min_value = 0.25, max_value = 4)
    filterOrderSlider = dpg.add_slider_int(label = "filter order", pos = (30, baseYPos + 180), default_value = 2, min_value = 1, max_value = 3)

with dpg.window(label = "Raw", tag = "raw_window", width = 515, height = 370, pos = (515,0)):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - rawHeight[i]), (10 + 15 * i, baseYPos - rawHeight[i]), color = (220, 220, 220), tag = f"dynamic_box_raw{i}")
    dpg.draw_text((10, baseYPos + 30), f"current device id: {DEVICE_ID}", color = (250, 250, 250, 255), size = 15)
    dpg.draw_text((10, baseYPos + 50), "frametime: 0ms", color = (250, 250, 250, 255), size = 15, tag = "frametime")
    fpsSlider = dpg.add_slider_int(label = "target fps", pos = (30, baseYPos + 105), default_value = 180, min_value = 15, max_value = 180)

with dpg.item_handler_registry(tag = "exit_handler") as handler:
    dpg.add_item_clicked_handler(callback = exit_program)

dpg.bind_item_handler_registry("exit_button", "exit_handler")

dpg.set_viewport_vsync = True

dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1045, height = 400)
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    startTime = time.time()
    fps = dpg.get_value(fpsSlider)
    frame()
    dpg.render_dearpygui_frame()
    time.sleep(1/fps)
    endTime = time.time()
    frametime = (endTime - startTime)  * 1000
    dpg.configure_item("frametime", text = f"frametime: {frametime:.0f}ms")