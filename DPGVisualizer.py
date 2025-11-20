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



### system
# init audio stream
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



### vars
# system
fps = 180
colorBandSize = 64

# audio analysis
useOrder = 1
bars = 32 
dataSizePerBox = (CHUNK / (2 * bars))
influnceMultipler = (1/ (64))
startFreq = 80
endFreq = 20000
high = startFreq
base = (endFreq / startFreq) ** (1 / bars)
referenceBar = 5

# bars
velocity = np.empty(bars, dtype = int)
rawHeight = np.full(bars, 1, dtype = int)
height = np.full(bars, 1, dtype = int)
gradientR = np.full(colorBandSize, 0, dtype = int)
gradientG = np.full(colorBandSize, 0, dtype = int)
gradientB = np.full(colorBandSize, 0, dtype = int)
exponentialDecay = 0.38
exponentBase = 100
decaySpeed = 0.15
powerMultiplier = 1.5
pumpThreshold = 22

# layers
layerPower = 0.5
layerMultiplier = 10
baseMultiplier = 0.3
base_rainbowSpeed = 0.15
base_rainbowScale = 0.45
base_rainbowSat = 0.6
layer_offset = 0.0
layer_lowsHue = 0.45
layer_lowsSat = 1.0
layer_centeredWaveHue = 0.5
layer_centeredWaveHueSpread = 0.015
layer_centeredWaveSat = 1

# colors
base_rainbowHue = 0.0
baseRGB = [(0,0,0)] * colorBandSize
layerRGB = [(0,0,0)] * colorBandSize
layerMask = [0] * colorBandSize

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



### functions
# audio processing
def bandpass_filter(data, lowcut, highcut, fs, order):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype = 'bandpass')
    y = lfilter(b, a, data)
    return y

def audio_process():
    global left

    while (running):
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
    for i in range(bars):
        # old, linear
        # if (rawHeight[i] > height[i]):

        if (rawHeight[i] - height[i] > pumpThreshold):
            velocity[i] = rawHeight[i] * powerMultiplier
        height[i] += velocity[i] * 0.5 * (180 / fps)
        height[i] -= exponentialDecay * height[i] * (180 / fps)
        if (height[i] <= 1 or height[i] > 1000):
            height[i] = 1
        else:
            velocity[i] -= (decaySpeed * 7.5 * (180 / fps))

def waveform_update():
    dpg.set_value("waveform_series", [shortWaveformx, shortWaveform])

# system
def frame():
    LED_update_base_rainbow()
    LED_update_layer_centered_wave()

    bars_update()
    waveform_update()
    LED_bake()
    simulated_LED_update()

    for i in range(bars):
       dpg.configure_item(f"height_text_raw{i}", text = str(rawHeight[i]))
       dpg.configure_item(f"height_text{i}", text = str(height[i]))
       dpg.configure_item(f"dynamic_box_raw{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - rawHeight[i]), p4 = (10 + 15 * i, baseYPos - rawHeight[i]))
       dpg.configure_item(f"dynamic_box{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - height[i]), p4 = (10 + 15 * i, baseYPos - height[i]))
    dpg.configure_item("window_size_text", text = f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15)

def exit_program():
    print("exiting program...")
    stream.stop_stream()
    dpg.destroy_context()
    quit()

def update_properties():
    global fps
    global useOrder
    global referenceBar
    global decaySpeed
    global powerMultiplier
    global exponentialDecay
    global exponentBase
    global pumpThreshold
    global layerPower
    global layerMultiplier
    global baseMultiplier
    global base_rainbowSpeed
    global base_rainbowScale
    global base_rainbowSat
    global layer_offset
    global layer_lowsHue
    global layer_lowsSat
    global layer_centeredWaveHue
    global layer_centeredWaveHueSpread
    global layer_centeredWaveSat

    fps = dpg.get_value(fpsSlider)
    useOrder = dpg.get_value(filterOrderSlider)
    referenceBar = dpg.get_value(referenceBarSlider)
    decaySpeed = dpg.get_value(decaySpeedSlider)
    powerMultiplier = dpg.get_value(powerMultiplierSlider)
    exponentialDecay = dpg.get_value(exponentialDecaySlider)
    exponentBase = dpg.get_value(exponentBaseSlider)
    pumpThreshold = dpg.get_value(pumpThresholdSlider)
    layerPower = dpg.get_value(layerPowerSlider)
    layerMultiplier = dpg.get_value(layerMultiplierSlider)
    baseMultiplier = dpg.get_value(baseMultiplierSlider)
    base_rainbowSpeed = dpg.get_value(base_rainbowSpeedSlider)
    base_rainbowScale = dpg.get_value(base_rainbowScaleSlider)
    base_rainbowSat = dpg.get_value(base_rainbowSatSlider)
    layer_offset = dpg.get_value(layer_offsetSlider)
    layer_lowsHue = dpg.get_value(layer_lowsHueSlider)
    layer_lowsSat = dpg.get_value(layer_lowsSatSlider)
    layer_centeredWaveHue = dpg.get_value(layer_centeredWaveHueSlider)
    layer_centeredWaveHueSpread = dpg.get_value(layer_centeredWaveHueSpreadSlider)
    layer_centeredWaveSat = dpg.get_value(layer_centeredWaveSatSlider)

    dpg.configure_item("reference_bar_line", p1 = (15 + 15 * referenceBar, baseYPos), p2 = (15 + 15 * referenceBar, 20))

## patterns
# base
def LED_update_base_rainbow():
    global base_rainbowHue

    for i in range(colorBandSize):
        baseRGB[i] = colorsys.hsv_to_rgb(base_rainbowHue + (i * base_rainbowScale / colorBandSize), base_rainbowSat, baseMultiplier * 255)

    base_rainbowHue += base_rainbowSpeed / 100

# layers
def LED_update_layer_lows():
    for i in range(colorBandSize):
        layerRGB[i] = colorsys.hsv_to_rgb(layer_lowsHue, layer_lowsSat, layerMultiplier * 255)
        layerMask[i] =  min(max((height[referenceBar] * layerPower - i + layer_offset) / 255, 0), 255)

def LED_update_layer_centered_wave():
    for i in range(int(colorBandSize / 2)):
        layerRGB[int(colorBandSize / 2) + i] = colorsys.hsv_to_rgb(layer_centeredWaveHue + (layer_centeredWaveHueSpread * i), layer_centeredWaveSat, layerMultiplier * 255)
        layerRGB[int(colorBandSize / 2) - i - 1] = colorsys.hsv_to_rgb(layer_centeredWaveHue + (layer_centeredWaveHueSpread * i), layer_centeredWaveSat, layerMultiplier * 255)
        layerMask[int(colorBandSize / 2) + i] =  min(max((height[referenceBar] * layerPower - i + layer_offset) / 255, 0), 255)
        layerMask[int(colorBandSize / 2) - i - 1] =  min(max((height[referenceBar] * layerPower - i + layer_offset) / 255, 0), 255)

# LED
def LED_bake():
    # base
    for i in range(colorBandSize):
        gradientR[i] = baseRGB[i][0]
        gradientG[i] = baseRGB[i][1]
        gradientB[i] = baseRGB[i][2]

    # layer
    for i in range(colorBandSize):
        gradientR[i] += layerRGB[i][0] * layerMask[i]
        gradientG[i] += layerRGB[i][1] * layerMask[i]
        gradientB[i] += layerRGB[i][2] * layerMask[i]

def simulated_LED_update():
    for i in range(60):
        dpg.configure_item(f"led60_index{i}", fill = (gradientR[int(i * colorBandSize / 60)], gradientG[int(i * colorBandSize / 60)], gradientB[int(i * colorBandSize / 60)]))
    for i in range(144):
        dpg.configure_item(f"led144_index{i}", fill = (gradientR[int(i * colorBandSize / 144)], gradientG[int(i * colorBandSize / 144)], gradientB[int(i * colorBandSize / 144)]))



### threading
t = threading.Thread(target = audio_process)
t.daemon = True
t.start()



### DPG UI
dpg.create_context()

# windows
with dpg.window(label = "Main", tag = "main_window", width = 630, height = 800, pos = (515,0)):
    dpg.draw_line((15 + 15 * referenceBar, baseYPos), (15 + 15 * referenceBar, 20), color = colorsys.hsv_to_rgb(0.2,0.8,200), tag = "reference_bar_line")
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(height[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - height[i]), (10 + 15 * i, baseYPos - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
    dpg.draw_text((10, baseYPos + 30), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
    dpg.add_button(pos = (30, baseYPos + 95), label = "exit", tag = "exit_action")
    
    decaySpeedSlider = dpg.add_slider_float(label = "decay speed", pos = (30, baseYPos + 120), default_value = decaySpeed, min_value = 0, max_value = 10, tag = "decay_speed_action")
    powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", pos = (30, baseYPos + 140), default_value = powerMultiplier, min_value = 0.25, max_value = 50, tag = "power_multiplier_action")
    exponentialDecaySlider = dpg.add_slider_float(label = "exponential decay", pos = (30, baseYPos + 160), default_value = exponentialDecay, min_value = 0, max_value = 1, tag = "exponential_decay_action")
    exponentBaseSlider = dpg.add_slider_float(label = "exponent base (unused)", pos = (30, baseYPos + 180), default_value = exponentBase, min_value = 0, max_value = 100, tag = "exponent_base_action")
    pumpThresholdSlider = dpg.add_slider_int(label = "pump threshold", pos = (30, baseYPos + 200), default_value = pumpThreshold, min_value = 10, max_value = 100, tag = "pump_threshold_action")

    filterOrderSlider = dpg.add_slider_int(label = "filter order", pos = (30, baseYPos + 240), default_value = useOrder, min_value = 1, max_value = 3, tag = "filter_order_action")
    referenceBarSlider = dpg.add_slider_int(label = "reference bar", pos = (30, baseYPos + 260), default_value = referenceBar, min_value = 0, max_value = bars - 1, tag = "reference_bar_action")

    anchor = baseYPos + 260
    dpg.draw_text((10, anchor), "1m x 60leds", size = 15)

    anchor += 25
    for i in range(60):
        dpg.draw_quad((30 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor), (30 + i * 7, anchor), fill = (0,0,0), tag = f"led60_index{i}", color = (0,0,0,0))

    anchor += 25
    dpg.draw_text((10, anchor), "1m x 144leds", size = 15)

    anchor += 25
    for i in range(144):
        dpg.draw_quad((30 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor), (30 + i * 3, anchor), fill = (0,0,0), tag = f"led144_index{i}", color = (0,0,0,0))

    anchor += 55
    layerPowerSlider = dpg.add_slider_float(label = "gradient power", pos = (30, anchor), default_value = layerPower, min_value = 0.1, max_value = 5, tag = "layer_power_action")
    layerMultiplierSlider = dpg.add_slider_float(label = "gradient multiplier", pos = (30, anchor + 20), default_value = layerMultiplier, min_value = 0.2, max_value = 10, tag = "layer_multiplier_action")
    layer_offsetSlider = dpg.add_slider_float(label = "gate offset", pos = (30, anchor + 40), default_value = layer_offset, min_value = -50, max_value = 50, tag = "layer_offset_action")

    anchor += 60
    layer_lowsHueSlider = dpg.add_slider_float(label = "lows hue", pos = (30, anchor), default_value = layer_lowsHue, min_value = 0, max_value = 1, tag = "layer_lows_hue_action")
    layer_lowsSatSlider = dpg.add_slider_float(label = "lows saturation", pos = (30, anchor + 20), default_value = layer_lowsSat, min_value = 0, max_value = 1, tag = "layer_lows_sat_action")
    layer_centeredWaveHueSlider = dpg.add_slider_float(label = "centered wave hue", pos = (30, anchor + 40), default_value = layer_centeredWaveHue, min_value = 0, max_value = 1, tag = "layer_centered_wave_hue_action")
    layer_centeredWaveHueSpreadSlider = dpg.add_slider_float(label = "centered wave hue spread", pos = (30, anchor + 60), default_value = layer_centeredWaveHueSpread, min_value = -0.1, max_value = 0.1, tag = "layer_centered_wave_hue_spread_action")
    layer_centeredWaveSatSlider = dpg.add_slider_float(label = "centered wave saturation", pos = (30, anchor + 80), default_value = layer_centeredWaveSat, min_value = 0, max_value = 1, tag = "layer_centered_wave_sat_action")

    anchor += 100
    base_rainbowSpeedSlider = dpg.add_slider_float(label = "base speed", pos = (30, anchor), default_value = 0.1, min_value = 0, max_value = 10, tag = "base_rainbow_speed_action")
    baseMultiplierSlider = dpg.add_slider_float(label = "base multiplier", pos = (30, anchor + 20), default_value = 0.3, min_value = 0, max_value = 1, tag = "base_multiplier_action")
    base_rainbowSatSlider = dpg.add_slider_float(label = "base saturation", pos = (30, anchor + 40), default_value = 0.6, min_value = 0, max_value = 1, tag = "base_rainbow_sat_action")
    base_rainbowScaleSlider = dpg.add_slider_float(label = "base gradient scale", pos = (30, anchor + 60), default_value = 0.45, min_value = 0, max_value = 2, tag = "base_rainbow_scale_action")

with dpg.window(label = "Raw", tag = "raw_window", width = 515, height = 800):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - rawHeight[i]), (10 + 15 * i, baseYPos - rawHeight[i]), color = (220, 220, 220), tag = f"dynamic_box_raw{i}")
    dpg.draw_text((10, baseYPos + 30), f"current device id: {DEVICE_ID}", color = (250, 250, 250, 255), size = 15)
    dpg.draw_text((10, baseYPos + 50), "frametime: 0ms", color = (250, 250, 250, 255), size = 15, tag = "frametime")
    fpsSlider = dpg.add_slider_int(label = "target fps", pos = (30, baseYPos + 105), default_value = 180, min_value = 15, max_value = 240)
    with dpg.plot(label = "waveform", height = 200, width = 455, pos = (30, baseYPos + 130)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label = "sample")
        dpg.add_plot_axis(dpg.mvYAxis, label = "", tag = "y_axis")
        dpg.set_axis_limits("y_axis", -32000, 32000)
        dpg.add_line_series(shortWaveformx, shortWaveform, label = "left channel", tag = "waveform_series", parent = "y_axis")

# handler

handles = {"exit": exit_program, 
           "decay_speed": update_properties,
           "power_multiplier": update_properties,
           "exponential_decay": update_properties,
           "exponent_base": update_properties,
           "pump_threshold": update_properties,
           "filter_order": update_properties,
           "reference_bar": update_properties,
           "layer_power": update_properties,
           "layer_multiplier": update_properties,
           "layer_offset": update_properties,
           "layer_lows_hue": update_properties,
           "layer_lows_sat": update_properties,
           "layer_centered_wave_hue": update_properties,
           "layer_centered_wave_hue_spread": update_properties,
           "layer_centered_wave_sat": update_properties,
           "base_rainbow_speed": update_properties,
           "base_multiplier": update_properties,
           "base_rainbow_sat": update_properties,
           "base_rainbow_scale": update_properties
           }

for handle in handles:
    with dpg.item_handler_registry(tag = f"{handle}_handler") as handler:
        dpg.add_item_active_handler(callback = handles[handle])
    dpg.bind_item_handler_registry(f"{handle}_action", f"{handle}_handler")

# draw ui
dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1155, height = 800 + 35)
dpg.setup_dearpygui()
dpg.show_viewport()

# frame update
while dpg.is_dearpygui_running():
    startTime = time.time()
    frame()
    dpg.render_dearpygui_frame()
    endTime = time.time()
    frametime = (endTime - startTime)
    time.sleep(abs((1/fps) - (frametime)))
    dpg.configure_item("frametime", text = f"process frametime: {frametime * 1000:.0f}ms")