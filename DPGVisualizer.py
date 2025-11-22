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
baseMode = 0
layerMode = 1
layerPower = 0.5
layerMultiplier = 1
baseMultiplier = 0.3
baseHue = 0.0
base_rainbowSpeed = 0.15
base_rainbowScale = 0.45
baseSat = 0.6
layerOffset = 0.0
layerHue = 0.5
layerSat = 1
layer_sideOrientation = 0
layerHueSpread = 0.015

# colors
base_rainbowHue = 0.0
baseRGB = [(0,0,0)] * colorBandSize
layerRGB = [(0,0,0)] * colorBandSize

# ui
baseYPos = 150
baseModeDict = {0: "Rainbow", 1: "Static", 2: "Breathing"}
layerModeDict = {0: "One Side", 1: "Middle", 2: "Pulse"}

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
        # old
        # if (rawHeight[i] > height[i]):

        if (rawHeight[i] - height[i] > pumpThreshold):
            velocity[i] = rawHeight[i] * powerMultiplier
        height[i] += velocity[i] * 0.5
        height[i] -= exponentialDecay * height[i]
        if (height[i] <= 1 or height[i] > 1000):
            height[i] = 1
        else:
            velocity[i] -= (decaySpeed * 7.5 * (180 / fps))

def waveform_update():
    dpg.set_value("waveform_series", [shortWaveformx, shortWaveform])

# system
def frame():
    match baseMode:
        case 0:
            LED_update_base_rainbow()
        case 1:
            LED_update_base_static()
    
    match layerMode:
        case 0:
            LED_update_layer_lows()
        case 1:
            LED_update_layer_centered_wave()
        case 2:
            LED_update_layer_pulse()

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
    global baseHue
    global base_rainbowSpeed
    global base_rainbowScale
    global baseSat
    global layerOffset
    global layerHue
    global layerSat
    global layer_sideOrientation
    global layerHueSpread

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
    baseHue = dpg.get_value(baseHueSlider)
    base_rainbowSpeed = dpg.get_value(base_rainbowSpeedSlider)
    base_rainbowScale = dpg.get_value(base_rainbowScaleSlider)
    baseSat = dpg.get_value(baseSatSlider)
    layerOffset = dpg.get_value(layerOffsetSlider)
    layerHue = dpg.get_value(layerHueSlider)
    layerSat = dpg.get_value(layerSatSlider)
    layerHueSpread = dpg.get_value(layerHueSpreadSlider)
    layer_sideOrientation = dpg.get_value(layer_sideOrientationSlider)

    dpg.configure_item("reference_bar_line", p1 = (15 + 15 * referenceBar, baseYPos), p2 = (15 + 15 * referenceBar, 20))

## patterns
# base
def LED_update_base_rainbow():
    global base_rainbowHue

    for i in range(colorBandSize):
        baseRGB[i] = colorsys.hsv_to_rgb(base_rainbowHue + (i * base_rainbowScale / colorBandSize) + baseHue, baseSat, baseMultiplier * 255)

    base_rainbowHue += base_rainbowSpeed / 100

def LED_update_base_static():
    for i in range(colorBandSize):
        baseRGB[i] = colorsys.hsv_to_rgb(baseHue, baseSat, baseMultiplier * 255)

# layers
def LED_update_layer_lows():
    match layer_sideOrientation:
        case 0:
            for i in range(colorBandSize):
                layerRGB[i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        case 2:    
            for i in range(colorBandSize):
                layerRGB[colorBandSize - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        case 1:
            for i in range(int(colorBandSize / 2)):
                layerRGB[i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 1275)
                layerRGB[colorBandSize - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 1275)

def LED_update_layer_centered_wave():
    for i in range(int(colorBandSize / 2)):
        layerRGB[int(colorBandSize / 2) + i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        layerRGB[int(colorBandSize / 2) - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)

def LED_update_layer_pulse():
    for i in range(int(colorBandSize / 2)):
        layerRGB[int(colorBandSize / 2) + i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        layerRGB[int(colorBandSize / 2) - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)

# LED
def LED_bake():
    # base
    for i in range(colorBandSize):
        gradientR[i] = baseRGB[i][0]
        gradientG[i] = baseRGB[i][1]
        gradientB[i] = baseRGB[i][2]

    # layer
    for i in range(colorBandSize):
        gradientR[i] += layerRGB[i][0]
        gradientG[i] += layerRGB[i][1]
        gradientB[i] += layerRGB[i][2]

def simulated_LED_update():
    for i in range(60):
        dpg.configure_item(f"led60_index{i}", fill = (gradientR[int(i * colorBandSize / 60)], gradientG[int(i * colorBandSize / 60)], gradientB[int(i * colorBandSize / 60)]))
    for i in range(144):
        dpg.configure_item(f"led144_index{i}", fill = (gradientR[int(i * colorBandSize / 144)], gradientG[int(i * colorBandSize / 144)], gradientB[int(i * colorBandSize / 144)]))

# ui
def ui_update_menu_items():
    global baseMode
    global layerMode

    baseMode = dpg.get_value(baseModeSlider)
    layerMode = dpg.get_value(layerModeSlider)

    ui_hide_all_mode_sliders()
    
    dpg.configure_item("layer_mode_text", default_value = f"layer mode: {layerModeDict[layerMode]}")
    dpg.configure_item("base_mode_text", default_value = f"base mode: {baseModeDict[baseMode]}")

    match baseMode:
        case 0:
            dpg.show_item("base_rainbow_speed_action")
            dpg.show_item("base_rainbow_scale_action")

    match layerMode:
        case 0:
            dpg.show_item("layer_side_orientation_action")
            dpg.show_item("layer_hue_spread_action")
        case 1:
            dpg.show_item("layer_hue_spread_action")

def ui_hide_all_mode_sliders():
    dpg.hide_item("base_rainbow_speed_action")
    dpg.hide_item("base_rainbow_scale_action")
    dpg.hide_item("layer_hue_spread_action")
    dpg.hide_item("layer_side_orientation_action")

### threading
t = threading.Thread(target = audio_process)
t.daemon = True
t.start()



### DPG UI
dpg.create_context()

# windows
with dpg.window(label = "Main", tag = "main_window", width = 1030, height = 600, pos = (515,0)):
    dpg.draw_line((15 + 15 * referenceBar, baseYPos), (15 + 15 * referenceBar, 20), color = colorsys.hsv_to_rgb(0.2,0.8,200), tag = "reference_bar_line")
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(height[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - height[i]), (10 + 15 * i, baseYPos - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
    dpg.draw_text((30, baseYPos + 40), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
    dpg.add_button(pos = (400, baseYPos + 66), label = "exit", tag = "exit_action")

    anchor = baseYPos + 100

    dpg.draw_text((30, anchor - 23), "1m x 60leds", size = 15)
    for i in range(60):
        dpg.draw_quad((30 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor + 10), (30 + 5 + i * 7, anchor), (30 + i * 7, anchor), fill = (0,0,0), tag = f"led60_index{i}", color = (0,0,0,0))
    
    anchor += 50

    dpg.draw_text((30, anchor - 23), "1m x 144leds", size = 15)
    for i in range(144):
        dpg.draw_quad((30 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor + 10), (30 + 1 + i * 3, anchor), (30 + i * 3, anchor), fill = (0,0,0), tag = f"led144_index{i}", color = (0,0,0,0))
    
    anchor += 75

    with dpg.child_window(tag = "main_parent_group_1", pos = (30, anchor), border = False, resizable_x = False, width = 515):
        dpg.add_text("bars settings")
        decaySpeedSlider = dpg.add_slider_float(label = "decay speed", default_value = decaySpeed, min_value = 0, max_value = 10, tag = "decay_speed_action")
        powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", default_value = powerMultiplier, min_value = 0.25, max_value = 50, tag = "power_multiplier_action")
        exponentialDecaySlider = dpg.add_slider_float(label = "exponential decay", default_value = exponentialDecay, min_value = 0, max_value = 1, tag = "exponential_decay_action")
        exponentBaseSlider = dpg.add_slider_float(label = "exponent base (unused)", default_value = exponentBase, min_value = 0, max_value = 100, tag = "exponent_base_action")
        pumpThresholdSlider = dpg.add_slider_int(label = "pump threshold", default_value = pumpThreshold, min_value = 10, max_value = 100, tag = "pump_threshold_action")
        filterOrderSlider = dpg.add_slider_int(label = "filter order", default_value = useOrder, min_value = 1, max_value = 3, tag = "filter_order_action")
        referenceBarSlider = dpg.add_slider_int(label = "reference bar", default_value = referenceBar, min_value = 0, max_value = bars - 1, tag = "reference_bar_action")

    anchor = 53

    with dpg.child_window(tag = "main_parent_group_2", pos = (515 + 35, anchor), border = False, resizable_x = False, width = 515):
        dpg.add_text("layer settings")
        layerPowerSlider = dpg.add_slider_float(label = "layer power", default_value = layerPower, min_value = 0.1, max_value = 2, tag = "layer_power_action")
        layerMultiplierSlider = dpg.add_slider_float(label = "layer multiplier", default_value = layerMultiplier, min_value = 0.2, max_value = 10, tag = "layer_multiplier_action")
        layerOffsetSlider = dpg.add_slider_float(label = "layer offset", default_value = layerOffset, min_value = -50, max_value = 50, tag = "layer_offset_action")
        layerHueSlider = dpg.add_slider_float(label = "layer hue", default_value = layerHue, min_value = 0, max_value = 1, tag = "layer_hue_action")
        layerSatSlider = dpg.add_slider_float(label = "layer saturation", default_value = layerSat, min_value = 0, max_value = 1, tag = "layer_sat_action")
        dpg.add_text("")
        dpg.add_text(f"layer mode: {layerModeDict[layerMode]}", tag = "layer_mode_text")
        layerModeSlider = dpg.add_slider_int(label = "layer mode", default_value = layerMode, min_value = 0, max_value = 2, tag = "layer_mode_action")
        layer_sideOrientationSlider = dpg.add_slider_int(label = "layer orientation", default_value = 0, min_value = 0, max_value = 2, tag = "layer_side_orientation_action")
        layerHueSpreadSlider = dpg.add_slider_float(label = "hue spread", default_value = layerHueSpread, min_value = -0.1, max_value = 0.1, tag = "layer_hue_spread_action")
        dpg.add_text("")
        dpg.add_text("base settings")
        baseMultiplierSlider = dpg.add_slider_float(label = "base multiplier", default_value = 0.3, min_value = 0, max_value = 1, tag = "base_multiplier_action")
        baseHueSlider = dpg.add_slider_float(label = "base hue", default_value = 0, min_value = 0, max_value = 1, tag = "base_hue_action")
        baseSatSlider = dpg.add_slider_float(label = "base saturation", default_value = 0.6, min_value = 0, max_value = 1, tag = "base_sat_action")
        dpg.add_text("")
        dpg.add_text(f"base mode: {baseModeDict[baseMode]}", tag = "base_mode_text")
        baseModeSlider = dpg.add_slider_int(label = "base mode", default_value = baseMode, min_value = 0, max_value = 2, tag = "base_mode_action")
        base_rainbowSpeedSlider = dpg.add_slider_float(label = "base speed", default_value = 0.1, min_value = 0, max_value = 10, tag = "base_rainbow_speed_action")
        base_rainbowScaleSlider = dpg.add_slider_float(label = "base gradient scale", default_value = 0.45, min_value = 0, max_value = 2, tag = "base_rainbow_scale_action")

    # base individual
    # anchor = dpg.get_item_pos("base_group")[1] + dpg.get_item_rect_size("base_group")[1]
    

with dpg.window(label = "Raw", tag = "raw_window", width = 515, height = 600):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - rawHeight[i]), (10 + 15 * i, baseYPos - rawHeight[i]), color = (220, 220, 220), tag = f"dynamic_box_raw{i}")
    dpg.draw_text((30, baseYPos + 30), f"current device id: {DEVICE_ID}", color = (250, 250, 250, 255), size = 15)
    dpg.draw_text((30, baseYPos + 50), "frametime: 0ms", color = (250, 250, 250, 255), size = 15, tag = "frametime")
    fpsSlider = dpg.add_slider_int(label = "target fps", pos = (30, baseYPos + 105), default_value = 180, min_value = 15, max_value = 240, tag = "fps_action")
    with dpg.plot(label = "waveform", height = 200, width = 455, pos = (30, baseYPos + 130)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label = "sample")
        dpg.add_plot_axis(dpg.mvYAxis, label = "", tag = "y_axis")
        dpg.set_axis_limits("y_axis", -32000, 32000)
        dpg.add_line_series(shortWaveformx, shortWaveform, label = "left channel", tag = "waveform_series", parent = "y_axis")

ui_update_menu_items()

# handler

handles = {"exit": exit_program, 
           "decay_speed": update_properties,
           "power_multiplier": update_properties,
           "exponential_decay": update_properties,
           "exponent_base": update_properties,
           "pump_threshold": update_properties,
           "filter_order": update_properties,
           "reference_bar": update_properties,
           "layer_mode": ui_update_menu_items,
           "layer_power": update_properties,
           "layer_multiplier": update_properties,
           "layer_offset": update_properties,
           "layer_hue": update_properties,
           "layer_sat": update_properties,
           "layer_side_orientation": update_properties,
           "layer_hue_spread": update_properties,
           "base_mode": ui_update_menu_items,
           "base_hue": update_properties,
           "base_rainbow_speed": update_properties,
           "base_multiplier": update_properties,
           "base_sat": update_properties,
           "base_rainbow_scale": update_properties,
           "fps": update_properties
           }

for handle in handles:
    with dpg.item_handler_registry(tag = f"{handle}_handler") as handler:
        dpg.add_item_active_handler(callback = handles[handle])
    dpg.bind_item_handler_registry(f"{handle}_action", f"{handle}_handler")

# draw ui
dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1545 + 15, height = 600 + 35, x_pos = 380, y_pos = 125)
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