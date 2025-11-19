import dearpygui.dearpygui as dpg
import time
import pyaudio
import numpy as np
import time
import threading
import colorsys
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, lfilter

# audio stream
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

# system
fps = 180
colorBand = 64

# audio analysis
bars = 32 
dataSizePerBox = (CHUNK / (2 * bars))
influnceMultipler = (1/ (64))
startFreq = 80
endFreq = 20000
high = startFreq
base = (endFreq / startFreq) ** (1 / bars)

# blocks
velocity = np.empty(bars, dtype = int)
rawHeight = np.full(bars, 1, dtype = int)
height = np.full(bars, 1, dtype = int)
gradientR = np.full(colorBand, 0, dtype = int)
gradientG = np.full(colorBand, 0, dtype = int)
gradientB = np.full(colorBand, 0, dtype = int)
# hueMap = 

# ui
baseYPos = 150

# flags
running = True

# monitoring
shortWaveform = [0] * int(CHUNK / 2)
shortWaveformx = []
for i in range(len(shortWaveform)):
    shortWaveformx.append(i)

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
    global left
    useOrder = 1

    while (running):
        try:
            useOrder = dpg.get_value(filterOrderSlider)
        except:
            pass
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            left = data[::2]
            for i in range(len(shortWaveform)):
                shortWaveform[i] = left[i].item()
            for i in range(bars):
                low = round(startFreq * base ** (i), 2)
                high = round(startFreq * base ** (i + 1), 2)
                rawHeight[i] = np.max(bandpass_filter(np.abs(left), low, high, RATE, useOrder)) * 1/64
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
        height[i] += velocity[i] * powerMultiplier * 0.075 * (180 / fps)
        if (height[i] <= 1 or height[i] > 1000):
            height[i] = 1
        else:
            velocity[i] -= decaySpeed * 7.5 * (180 / fps)

t = threading.Thread(target = audio_process)
t.daemon = True
t.start()

# DPG

dpg.create_context()

def frame():
    fps = dpg.get_value(fpsSlider)

    gradient_update()
    bars_update()
    waveform_update()
    simulated_LED_update()

    for i in range(bars):
       dpg.configure_item(f"height_text_raw{i}", text = str(rawHeight[i]))
       dpg.configure_item(f"height_text{i}", text = str(height[i]))
       dpg.configure_item(f"dynamic_box_raw{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - rawHeight[i]), p4 = (10 + 15 * i, baseYPos - rawHeight[i]))
       dpg.configure_item(f"dynamic_box{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - height[i]), p4 = (10 + 15 * i, baseYPos - height[i]))
    dpg.configure_item("window_size_text", text = f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15)

def waveform_update():
    dpg.set_value("waveform_series", [shortWaveformx, shortWaveform])

def gradient_update():
    gradientPower = dpg.get_value(gradientPowerSlider)
    gradientMultiplier = dpg.get_value(gradientMultiplierSlider)
    offset = dpg.get_value(offsetSlider) ##
    for i in range(colorBand):
        # color = colorsys.hsv_to_rgb(hueMap)
        gradientR[i] = max(height[4] * gradientPower - i + offset, 0) * gradientMultiplier
        gradientG[i] = max(height[4] * gradientPower - i + offset, 0) * gradientMultiplier
        gradientB[i] = max(height[4] * gradientPower - i + offset, 0) * gradientMultiplier

def simulated_LED_update():
    for i in range(60):
        dpg.configure_item(f"led60_index{i}", fill = (gradientR[int(i * colorBand / 60)], gradientG[int(i * colorBand / 60)], gradientB[int(i * colorBand / 60)]))
    for i in range(144):
        dpg.configure_item(f"led144_index{i}", fill = (gradientR[int(i * colorBand / 144)], gradientG[int(i * colorBand / 144)], gradientB[int(i * colorBand / 144)]))

def exit_program():
    print("exiting program...")
    stream.stop_stream()
    dpg.destroy_context()
    quit()

# windows
with dpg.window(label = "Main", tag = "main_window", width = 515, height = baseYPos + 450, pos = (515,0)):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(height[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - height[i]), (10 + 15 * i, baseYPos - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
    dpg.draw_text((10, baseYPos + 40), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
    dpg.add_button(pos = (30, baseYPos + 105), label = "exit", tag = "exit_button")
    decaySpeedSlider = dpg.add_slider_float(label = "decay speed", pos = (30, baseYPos + 130), default_value = 1, min_value = 0.25, max_value = 4)
    powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", pos = (30, baseYPos + 150), default_value = 1, min_value = 0.25, max_value = 4)
    filterOrderSlider = dpg.add_slider_int(label = "filter order", pos = (30, baseYPos + 190), default_value = 2, min_value = 1, max_value = 3)

    anchor = baseYPos + 210
    dpg.draw_text((10, anchor), "1m x 60leds", size = 15)

    anchor = baseYPos + 235
    for i in range(60):
        dpg.draw_quad((30 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor), (30 + i * 7, anchor), fill = (0,0,0), tag = f"led60_index{i}", color = (0,0,0,0))

    anchor = baseYPos + 260
    dpg.draw_text((10, anchor), "1m x 144leds", size = 15)

    anchor = baseYPos + 285
    for i in range(144):
        dpg.draw_quad((30 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor), (30 + i * 3, anchor), fill = (0,0,0), tag = f"led144_index{i}", color = (0,0,0,0))

    anchor = baseYPos + 340
    gradientPowerSlider = dpg.add_slider_float(label = "gradient power", pos = (30, anchor), default_value = 1, min_value = 0.2, max_value = 20)
    gradientMultiplierSlider = dpg.add_slider_float(label = "gradient multiplier", pos = (30, anchor + 20), default_value = 4, min_value = 0.2, max_value = 10)
    offsetSlider = dpg.add_slider_float(label = "gate offset", pos = (30, anchor + 40), default_value = 0, min_value = -50, max_value = 50)

with dpg.window(label = "Raw", tag = "raw_window", width = 515, height = baseYPos + 450):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - rawHeight[i]), (10 + 15 * i, baseYPos - rawHeight[i]), color = (220, 220, 220), tag = f"dynamic_box_raw{i}")
    dpg.draw_text((10, baseYPos + 30), f"current device id: {DEVICE_ID}", color = (250, 250, 250, 255), size = 15)
    dpg.draw_text((10, baseYPos + 50), "frametime: 0ms", color = (250, 250, 250, 255), size = 15, tag = "frametime")
    fpsSlider = dpg.add_slider_int(label = "target fps", pos = (30, baseYPos + 105), default_value = 180, min_value = 15, max_value = 240)
    with dpg.plot(label = "waveform", height = 200, width = 455, pos = (30, baseYPos + 135)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label = "sample")
        dpg.add_plot_axis(dpg.mvYAxis, label = "", tag = "y_axis")
        dpg.set_axis_limits("y_axis", -32000, 32000)
        dpg.add_line_series(shortWaveformx, shortWaveform, label = "left channel", tag = "waveform_series", parent = "y_axis")

# handler
with dpg.item_handler_registry(tag = "exit_handler") as handler:
    dpg.add_item_clicked_handler(callback = exit_program)

dpg.bind_item_handler_registry("exit_button", "exit_handler")

# draw ui
dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1045, height = baseYPos + 450 + 35)
dpg.setup_dearpygui()
dpg.show_viewport()

while dpg.is_dearpygui_running():
    startTime = time.time()
    fps = dpg.get_value(fpsSlider)
    frame()
    dpg.render_dearpygui_frame()
    endTime = time.time()
    frametime = (endTime - startTime)
    time.sleep(abs((1/fps) - (frametime)))
    dpg.configure_item("frametime", text = f"process frametime: {frametime * 1000:.0f}ms")