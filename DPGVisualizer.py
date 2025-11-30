import dearpygui.dearpygui as dpg
import time
import pyaudio
import sounddevice as sd
import numpy as np
import time
import threading
import colorsys
import serial
import cv2
from windows_capture import WindowsCapture, Frame, InternalCaptureControl
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import butter, lfilter



### system
# init audio stream
CHUNK = 512
RATE = 48000
DEVICE_ID = 50 # 3: mme, 22: wds, 50: wasapi
COLOR_BAND_SIZE = 64

LED_COUNT = 144

SCREEN_WIDTH = 3840
SCREEN_HEIGHT = 2160

pa = pyaudio.PyAudio()

stream = pa.open(
    format=pyaudio.paInt16, 
    channels=1, 
    rate=RATE, 
    input=True, 
    input_device_index=DEVICE_ID,
    frames_per_buffer=CHUNK, 
    )

stream.stop_stream()
if not stream.is_active():
    stream.start_stream()



### vars
# system
fps = 60
bakeMode = 0
captureBand = 12
captureBlocks = 24
captureStatus = 0
captureX = np.array([i / captureBand for i in range(captureBand)])
captureXBand = np.array([i / COLOR_BAND_SIZE for i in range(COLOR_BAND_SIZE)])
colorContributionThreshold = 0.33
colorContributionRatio = 0.7
colorValueThreshold = 32
colorValueCut = 0.85
colorTransitionSpeed = 1.0
colorCalibrationR = 0.96
colorCalibrationG = 1.0
colorCalibrationB = 0.91

# audio analysis
useOrder = 1
bars = 32 
dataSizePerBox = (CHUNK / (2 * bars))
influnceMultipler = (1/ (64))
startFreq = 80
endFreq = 20000
high = startFreq
base = (endFreq / startFreq) ** (1 / bars)
referenceBar = 1
a = [None] * bars
b = [None] * bars

# bars
velocity = np.empty(bars, dtype = int)
rawHeight = np.full(bars, 1, dtype = int)
rawHeightPrev = rawHeight.copy()
height = np.full(bars, 1, dtype = int)
gradientR = np.full(COLOR_BAND_SIZE, 0, dtype = int)
gradientG = np.full(COLOR_BAND_SIZE, 0, dtype = int)
gradientB = np.full(COLOR_BAND_SIZE, 0, dtype = int)
exponentialDecay = 0.08
exponentBase = 125
decaySpeed = 0.16
powerMultiplier = 0.875
powerOffset = 0
pumpThreshold = 12
pumpHeightThreshold = 0

# layers
baseMode = 1
layerMode = 0
layerPower = 0.25
layerMultiplier = 6 
layerOpacity = 1.0
baseMultiplier = 0
baseHue = 0.0
base_rainbowSpeed = 0.15
base_rainbowScale = 1
base_rainbowStart = 0.0
base_rainbowEnd = 1.0
base_breathingSpeed = 1.0
baseSat = 0.6
layerOffset = 0
layerHue = 0.5
layerSat = 1
layer_sideOrientation = 1
layer_pulseState = False
layer_pulseThreshold = 10
layer_pulseDecayRate = 0.1
layerHueSpread = 0.015
ledPatternSat = 1.0
ledPatternBrightness = 1.0
preset = 0

# colors
base_rainbowHue = 0.0
baseRGB = [(0,0,0)] * COLOR_BAND_SIZE
baseR = [0] * COLOR_BAND_SIZE
baseG = [0] * COLOR_BAND_SIZE
baseB = [0] * COLOR_BAND_SIZE
layerRGB = [(0,0,0)] * COLOR_BAND_SIZE
layerMask = [0.0] * COLOR_BAND_SIZE
columnAverageColor = [[0,0,0] for _ in range(captureBand)]

# ui
baseYPos = 150
baseModeDict = {0: "Rainbow", 1: "Static", 2: "Breathing"}
layerModeDict = {0: "One Side", 1: "Middle", 2: "Pulse"}
bakeModeDict = {0: "Audio", 1: "Screen Capture"}

# flags
running = True

# monitoring
shortWaveform = [0] * int(CHUNK / 2)
shortWaveformx = []
for i in range(len(shortWaveform)):
    shortWaveformx.append(i)

# LED
ledSerial = serial.Serial("COM9", 460800)
ledVignetteMultiplier = 0.0
ledVignettePower = 0.0



## screen capture init
capture = WindowsCapture(
    cursor_capture = None,
    draw_border = False,
    monitor_index = 1,
    window_name = None,
)

@capture.event
def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
    global captureStatus
    global baseR
    global baseG
    global baseB  

    if (captureStatus == 0):
        capture_control.stop()

    else:
        frameBuffer = frame.frame_buffer.view()

        for i in range(captureBand):
            columnAverageColor[i][0] = 0
            columnAverageColor[i][1] = 0
            columnAverageColor[i][2] = 0

        for i in range(captureBand):
            activeBlocks = captureBlocks
            
            for j in range(captureBlocks):
                columnAverageColorR = int(frameBuffer[j * int(SCREEN_HEIGHT / captureBlocks)][i * int(SCREEN_WIDTH / captureBand)][0])
                columnAverageColorG = int(frameBuffer[j * int(SCREEN_HEIGHT / captureBlocks)][i * int(SCREEN_WIDTH / captureBand)][1])
                columnAverageColorB = int(frameBuffer[j * int(SCREEN_HEIGHT / captureBlocks)][i * int(SCREEN_WIDTH / captureBand)][2])

                if ((max(columnAverageColorR, columnAverageColorG, columnAverageColorB) - min(columnAverageColorR, columnAverageColorG, columnAverageColorB)) / max(columnAverageColorR, columnAverageColorG, columnAverageColorB, 1) < colorContributionThreshold):
                    columnAverageColorR *= colorContributionRatio
                    columnAverageColorG *= colorContributionRatio
                    columnAverageColorB *= colorContributionRatio
                
                if (((columnAverageColorR + columnAverageColorG + columnAverageColorB) / 3 < colorValueThreshold) and (activeBlocks > colorValueCut)):
                    activeBlocks -= colorValueCut

                columnAverageColor[i][0] += columnAverageColorR
                columnAverageColor[i][1] += columnAverageColorG
                columnAverageColor[i][2] += columnAverageColorB

            columnAverageColor[i][0] /= activeBlocks
            columnAverageColor[i][1] /= activeBlocks
            columnAverageColor[i][2] /= activeBlocks

            # (shadow boost)
            columnGamma = pow((columnAverageColor[i][0] + columnAverageColor[i][1] + columnAverageColor[i][2]) / (3 * 255), 0.4)

            columnAverageColor[i][0] *= (1 + columnGamma)
            columnAverageColor[i][1] *= (1 + columnGamma)
            columnAverageColor[i][2] *= (1 + columnGamma)

    baseR = np.interp(captureXBand, captureX, np.array([columnAverageColor[i][2] for i in range(captureBand)]))
    baseG = np.interp(captureXBand, captureX, np.array([columnAverageColor[i][1] for i in range(captureBand)]))
    baseB = np.interp(captureXBand, captureX, np.array([columnAverageColor[i][0] for i in range(captureBand)]))

    time.sleep(2 / fps)

@capture.event
def on_closed():
    global captureStatus

    captureStatus = 0



### functions
# audio processing
def bandpass_coefficients():
    global a, b

    nyquist = 0.5 * RATE
    for i in range(bars):
        low = round(startFreq * base ** (i), 2) / nyquist
        high = round(startFreq * base ** (i + 1), 2) / nyquist
        b[i], a[i] = butter(useOrder, [low, high], btype = 'bandpass')

def bandpass_filter(data, index):
    # nyquist = 0.5 * fs
    # low = lowcut / nyquist
    # high = highcut / nyquist
    # b, a = butter(order, [low, high], btype = 'bandpass')
    y = lfilter(b[index], a[index], data)
    return y

def audio_process():
    global left

    rawHeightPrev = rawHeight.copy()
    
    while (running):
        try:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            left = data
            for i in range(len(shortWaveform)):
                shortWaveform[i] = left[i].item()
            for i in range(bars):
                rawHeight[i] = np.max(bandpass_filter(np.abs(left), i)) / 64
        except:
            pass

def bars_update():
    for i in range(bars):
        # ver1 compare
        # if (rawHeight[i] > height[i]):
        
        # ver2 delta smoothened 
        # if (rawHeight[i] - height[i] > pumpThreshold):
        
        # ver3 delta raw
        if ((rawHeight[i] - rawHeightPrev[i] > pumpThreshold) and rawHeight[i] > pumpHeightThreshold):
            velocity[i] = rawHeight[i] * powerMultiplier + powerOffset

        height[i] += velocity[i]
        height[i] -= exponentialDecay * height[i] / powerMultiplier

        if (height[i] <= 1 or height[i] > 1000):
            height[i] = 1
        else:
            velocity[i] -= (decaySpeed * 7.5 * (180 / fps))

def waveform_update():
    dpg.set_value("waveform_series", [shortWaveformx, shortWaveform])

# system
def frame():
    match bakeMode:
        case 0:
            match baseMode:
                case 0:
                    LED_update_base_rainbow()
                case 1:
                    LED_update_base_static()
                case 2:
                    LED_update_base_breathing()
            
            match layerMode:
                case 0:
                    LED_update_layer_lows()
                case 1:
                    LED_update_layer_centered_wave()
                case 2:
                    LED_update_layer_pulse()

            bars_update()
            waveform_update()

        case 1:
            LED_update_base_screen()
    
    LED_bake()
    simulated_LED_update()
    LED_update()

    for i in range(bars):
       dpg.configure_item(f"height_text_raw{i}", text = str(rawHeight[i]))
       dpg.configure_item(f"height_text{i}", text = str(height[i]))
       dpg.configure_item(f"dynamic_box_raw{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - rawHeight[i]), p4 = (10 + 15 * i, baseYPos - rawHeight[i]))
       dpg.configure_item(f"dynamic_box{i}", p1 = (10 + 15 * i, baseYPos), p2 = (20 + 15 * i, baseYPos), p3 = (20 + 15 * i, baseYPos - height[i]), p4 = (10 + 15 * i, baseYPos - height[i]))
    dpg.configure_item("window_size_text", text = f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15)

def exit_program():
    global captureStatus

    print("exiting program...")

    stream.stop_stream()
    dpg.destroy_context()
    quit()

def update_properties():
    global bakeMode
    global fps
    global useOrder
    global referenceBar
    global decaySpeed
    global powerMultiplier
    global powerOffset
    global exponentialDecay
    global pumpThreshold
    global pumpHeightThreshold
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
    global layerOpacity
    global layer_sideOrientation
    global layer_pulseThreshold
    global layer_pulseDecayRate
    global layerHueSpread
    global base_rainbowStart
    global base_rainbowEnd
    global base_breathingSpeed
    global ledPatternSat
    global ledVignettePower
    global ledVignetteMultiplier
    global ledPatternBrightness
    global colorContributionThreshold
    global colorContributionRatio
    global colorValueThreshold
    global colorValueCut
    global colorTransitionSpeed
    global colorCalibrationR
    global colorCalibrationG
    global colorCalibrationB

    bakeMode = dpg.get_value(bakeModeSlider)
    fps = dpg.get_value(fpsSlider)
    useOrder = dpg.get_value(filterOrderSlider)
    referenceBar = dpg.get_value(referenceBarSlider)
    decaySpeed = dpg.get_value(decaySpeedSlider)
    powerMultiplier = dpg.get_value(powerMultiplierSlider)
    powerOffset = dpg.get_value(powerOffsetSlider)
    exponentialDecay = dpg.get_value(exponentialDecaySlider)
    pumpThreshold = dpg.get_value(pumpThresholdSlider)
    pumpHeightThreshold = dpg.get_value(pumpHeightThresholdSlider)
    layerPower = dpg.get_value(layerPowerSlider)
    layerMultiplier = dpg.get_value(layerMultiplierSlider)
    baseMultiplier = dpg.get_value(baseMultiplierSlider)
    baseHue = dpg.get_value(baseHueSlider)
    base_rainbowSpeed = dpg.get_value(base_rainbowSpeedSlider)
    base_rainbowScale = dpg.get_value(base_rainbowScaleSlider)
    base_breathingSpeed = dpg.get_value(base_breathingSpeedSlider)
    baseSat = dpg.get_value(baseSatSlider)
    layerOffset = dpg.get_value(layerOffsetSlider)
    layerHue = dpg.get_value(layerHueSlider)
    layerSat = dpg.get_value(layerSatSlider)
    layerOpacity = dpg.get_value(layerOpacitySlider)
    layerHueSpread = dpg.get_value(layerHueSpreadSlider)
    layer_sideOrientation = dpg.get_value(layer_sideOrientationSlider)
    layer_pulseThreshold = dpg.get_value(layer_pulseThresholdSlider)
    layer_pulseDecayRate = dpg.get_value(layer_pulseDecayRateSlider)
    base_rainbowStart = dpg.get_value(base_rainbowStartSlider)
    base_rainbowEnd = dpg.get_value(base_rainbowEndSlider)
    ledVignettePower = dpg.get_value(ledVignettePowerSlider)
    ledVignetteMultiplier = dpg.get_value(ledVignetteMultiplierSlider)
    colorContributionThreshold = dpg.get_value(colorContributionThresholdSlider)
    colorContributionRatio = dpg.get_value(colorContributionRatioSlider)
    colorValueThreshold = dpg.get_value(colorValueThresholdSlider)
    colorValueCut = dpg.get_value(colorValueCutSlider)
    colorTransitionSpeed = dpg.get_value(colorTransitionSpeedSlider)
    colorCalibrationR = dpg.get_value(colorCalibrationRSlider)
    colorCalibrationG = dpg.get_value(colorCalibrationGSlider)
    colorCalibrationB = dpg.get_value(colorCalibrationBSlider)

    dpg.configure_item("reference_bar_line", p1 = (15 + 15 * referenceBar, baseYPos), p2 = (15 + 15 * referenceBar, 20))
    dpg.configure_item("reference_bar_pulse_threshold_line", p1 = (10 + 15 * referenceBar, baseYPos - layer_pulseThreshold), p2 = (20 + 15 * referenceBar, baseYPos - layer_pulseThreshold))

    bandpass_coefficients()

def update_bake_mode():
    global captureStatus

    update_properties()

    if (bakeMode == 1 and captureStatus == 0):
        captureStatus = 1
        capture.start_free_threaded()

    elif (bakeMode == 0):
        captureStatus = 0



# patterns
# base
def LED_update_base_rainbow():
    global base_rainbowHue

    # for i in range(colorBandSize):
    #     baseRGB[i] = colorsys.hsv_to_rgb(base_rainbowHue + (i * base_rainbowScale / colorBandSize) + baseHue, baseSat, baseMultiplier * 255)

    base_rainbowHue += base_rainbowSpeed

    if (base_rainbowStart == 0 and base_rainbowEnd == 1):
        for i in range(COLOR_BAND_SIZE):
            baseRGB[i] = colorsys.hsv_to_rgb(base_rainbowStart + ((base_rainbowEnd - base_rainbowStart) * ((i + base_rainbowHue) * base_rainbowScale / COLOR_BAND_SIZE)), baseSat, 255 * baseMultiplier)
    else:
        for i in range(COLOR_BAND_SIZE):
            baseRGB[i] = colorsys.hsv_to_rgb(base_rainbowStart + (((base_rainbowEnd - base_rainbowStart) * abs(((2 * ((i + base_rainbowHue) * base_rainbowScale / COLOR_BAND_SIZE)) % 2) - 1))), baseSat, 255 * baseMultiplier)

def LED_update_base_static():
    for i in range(COLOR_BAND_SIZE):
        baseRGB[i] = colorsys.hsv_to_rgb(baseHue, baseSat, baseMultiplier * 255)

def LED_update_base_screen():
    for i in range(COLOR_BAND_SIZE):
        # print(baseR[i], baseG[i], baseB[i])
        ### todo interp #######################################################################
        # baseRGB[i] = (int(columnAverageColor[int(i * captureBand / COLOR_BAND_SIZE)][0]), int(columnAverageColor[int(i * captureBand / COLOR_BAND_SIZE)][1]), int(columnAverageColor[int(i * captureBand / COLOR_BAND_SIZE)][2]))
        baseRGB[i] = (baseRGB[i][0] + ((baseR[i] - baseRGB[i][0]) * 3 * colorTransitionSpeed / fps), baseRGB[i][1] + ((baseG[i] - baseRGB[i][1]) * 3 * colorTransitionSpeed / fps), (baseRGB[i][2] + (baseB[i] - baseRGB[i][2]) * 3 * colorTransitionSpeed / fps))

def LED_update_base_breathing():
    for i in range(COLOR_BAND_SIZE):
        baseRGB[i] = colorsys.hsv_to_rgb(baseHue, baseSat, baseMultiplier * 255 * (np.cos(time.time() * base_breathingSpeed) + 1) / 2)

# presets
# base rainbow
def LED_update_base_preset():
    preset = dpg.get_value("preset_action")

    match preset:
        case "Purple Rainbow":
            dpg.set_value("base_mode_action", 0)
            dpg.set_value("base_rainbow_start_action", 0.75)
            dpg.set_value("base_rainbow_end_action", 0.95)
            dpg.set_value("base_sat_action", 0.8)
        case "Purple Blue Rainbow":
            dpg.set_value("base_mode_action", 0)
            dpg.set_value("base_rainbow_start_action", 0.55)
            dpg.set_value("base_rainbow_end_action", 0.85)
            dpg.set_value("base_sat_action", 0.8)
        case "Neon Rainbow":
            dpg.set_value("base_mode_action", 0)
            dpg.set_value("base_rainbow_start_action", 0.35)
            dpg.set_value("base_rainbow_end_action", 0.65)
            dpg.set_value("base_sat_action", 0.9)
        case "Redshift Rainbow":
            dpg.set_value("base_mode_action", 0)
            dpg.set_value("base_rainbow_start_action", 0.0)
            dpg.set_value("base_rainbow_end_action", 0.135)
            dpg.set_value("base_sat_action", 0.95)
        case "Torch":
            dpg.set_value("base_mode_action", 1)
            dpg.set_value("base_multiplier_action", 1.0)
            dpg.set_value("base_hue_action", 0.056)
            dpg.set_value("base_sat_action", 1.0)
        case "Warm":
            dpg.set_value("base_mode_action", 1)
            dpg.set_value("base_multiplier_action", 1.0)
            dpg.set_value("base_hue_action", 0.09)
            dpg.set_value("base_sat_action", 0.95)
        case "Calm":
            dpg.set_value("base_mode_action", 1)
            dpg.set_value("base_multiplier_action", 1.0)
            dpg.set_value("base_hue_action", 0.115)
            dpg.set_value("base_sat_action", 0.8)

    ui_update_menu_items()
    update_properties()

# layers
def LED_update_layer_lows():
    match layer_sideOrientation:
        case 0:
            for i in range(COLOR_BAND_SIZE):
                layerRGB[i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        case 2:    
            for i in range(COLOR_BAND_SIZE):
                layerRGB[COLOR_BAND_SIZE - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        case 1:
            for i in range(int(COLOR_BAND_SIZE / 2)):
                layerRGB[i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 1275)
                layerRGB[COLOR_BAND_SIZE - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 1275)
    for i in range(COLOR_BAND_SIZE):
        layerMask[i] = 1.0

def LED_update_layer_centered_wave():
    for i in range(int(COLOR_BAND_SIZE / 2)):
        layerRGB[int(COLOR_BAND_SIZE / 2) + i] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
        layerRGB[int(COLOR_BAND_SIZE / 2) - i - 1] = colorsys.hsv_to_rgb(layerHue + (layerHueSpread * i), layerSat, min(max((height[referenceBar] * layerPower - i + layerOffset) / 255, 0), 255) * layerMultiplier * 2550)
    for i in range(COLOR_BAND_SIZE):
        layerMask[i] = 1.0

def LED_update_layer_pulse():
    global layer_pulseState

    for i in range(COLOR_BAND_SIZE):
        layerRGB[i] = colorsys.hsv_to_rgb(layerHue, layerSat, layerMultiplier * 255)

    for i in range(COLOR_BAND_SIZE - 1):
        layerMask[COLOR_BAND_SIZE - 1 - i] = layerMask[COLOR_BAND_SIZE - 2 - i]

    if ((layer_pulseState == False) and height[referenceBar] >= layer_pulseThreshold):
        layer_pulseState = True
        layerMask[0] = 1.0

    else:
        layerMask[0] -= layer_pulseDecayRate * layerMask[0]
        if (height[referenceBar] < layer_pulseThreshold):
            layer_pulseState = False

# LED
def LED_bake():
    # base
    for i in range(COLOR_BAND_SIZE):
        gradientR[i] = baseRGB[i][0]
        gradientG[i] = baseRGB[i][1]
        gradientB[i] = baseRGB[i][2]

    # layer
    for i in range(COLOR_BAND_SIZE):
        gradientR[i] += layerRGB[i][0] * layerMask[i] * layerOpacity
        gradientG[i] += layerRGB[i][1] * layerMask[i] * layerOpacity
        gradientB[i] += layerRGB[i][2] * layerMask[i] * layerOpacity

    # vignette
    for i in range(COLOR_BAND_SIZE):
        gradientR[i] = gradientR[i] * (1 - min(abs(((COLOR_BAND_SIZE / 2) - i) * ledVignettePower) * ledVignetteMultiplier / (COLOR_BAND_SIZE / 2), 1))
        gradientG[i] = gradientG[i] * (1 - min(abs(((COLOR_BAND_SIZE / 2) - i) * ledVignettePower) * ledVignetteMultiplier / (COLOR_BAND_SIZE / 2), 1))
        gradientB[i] = gradientB[i] * (1 - min(abs(((COLOR_BAND_SIZE / 2) - i) * ledVignettePower) * ledVignetteMultiplier / (COLOR_BAND_SIZE / 2), 1))

    # cap, calibration
    for i in range(COLOR_BAND_SIZE):
        gradientR[i] = min(max(int(gradientR[i]), 0), 255) * colorCalibrationR
        gradientG[i] = min(max(int(gradientG[i]), 0), 255) * colorCalibrationG
        gradientB[i] = min(max(int(gradientB[i]), 0), 255) * colorCalibrationB

def simulated_LED_update():
    for i in range(60):
        dpg.configure_item(f"led60_index{i}", fill = (gradientR[int(i * COLOR_BAND_SIZE / 60)], gradientG[int(i * COLOR_BAND_SIZE / 60)], gradientB[int(i * COLOR_BAND_SIZE / 60)]))

    for i in range(144):
        dpg.configure_item(f"led144_index{i}", fill = (gradientR[int(i * COLOR_BAND_SIZE / 144)], gradientG[int(i * COLOR_BAND_SIZE / 144)], gradientB[int(i * COLOR_BAND_SIZE / 144)]))

def LED_update():
    frame = bytearray()
    for i in range(LED_COUNT):
        frame += bytes([gradientR[int(i * COLOR_BAND_SIZE / LED_COUNT)], gradientG[int(i * COLOR_BAND_SIZE / LED_COUNT)], gradientB[int(i * COLOR_BAND_SIZE / LED_COUNT)]])
    ledSerial.write(frame)

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
            dpg.show_item("base_rainbow_start_action")
            dpg.show_item("base_rainbow_end_action")
        case 2:
            dpg.show_item("base_breathing_speed_action")

    match layerMode:
        case 0:
            dpg.show_item("layer_side_orientation_action")
            dpg.show_item("layer_hue_spread_action")
        case 1:
            dpg.show_item("layer_hue_spread_action")
        case 2:
            dpg.show_item("layer_pulse_threshold_action")
            dpg.show_item("layer_pulse_decay_rate_action")
            dpg.show_item("reference_bar_pulse_threshold_line")

def ui_hide_all_mode_sliders():
    dpg.hide_item("base_rainbow_speed_action")
    dpg.hide_item("base_rainbow_scale_action")
    dpg.hide_item("base_breathing_speed_action")
    dpg.hide_item("layer_hue_spread_action")
    dpg.hide_item("layer_side_orientation_action")
    dpg.hide_item("layer_pulse_threshold_action")
    dpg.hide_item("layer_pulse_decay_rate_action")
    dpg.hide_item("reference_bar_pulse_threshold_line")
    dpg.hide_item("base_rainbow_start_action")
    dpg.hide_item("base_rainbow_end_action")



### calculate bandpass coefficients
bandpass_coefficients()



### threading
t = threading.Thread(target = audio_process)
t.daemon = True
t.start()



### DPG UI
dpg.create_context()

# windows
with dpg.window(label = "Main", tag = "main_window", width = 1030, height = 715, pos = (515,0)):
    dpg.draw_line((15 + 15 * referenceBar, baseYPos), (15 + 15 * referenceBar, 20), color = colorsys.hsv_to_rgb(0.15,0.8,200), tag = "reference_bar_line")
    dpg.draw_line((10 + 15 * referenceBar, baseYPos - layer_pulseThreshold), (20 + 15 * referenceBar, baseYPos - layer_pulseThreshold), color = colorsys.hsv_to_rgb(0.15,0.8,200), tag = "reference_bar_pulse_threshold_line")
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
        dpg.add_text(f"bake mode: {bakeModeDict[bakeMode]}", tag = "bake_mode_text")
        bakeModeSlider = dpg.add_slider_int(label = "bake mode", default_value = bakeMode, min_value = 0, max_value = 1, tag = "bake_mode_action")
        dpg.add_text("")
        dpg.add_text("bars settings")
        decaySpeedSlider = dpg.add_slider_float(label = "decay speed", default_value = decaySpeed, min_value = 0, max_value = 10, tag = "decay_speed_action")
        powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", default_value = powerMultiplier, min_value = 0, max_value = 50, tag = "power_multiplier_action")
        powerOffsetSlider = dpg.add_slider_float(label = "power offset", default_value = powerOffset, min_value = -50, max_value = 50, tag = "power_offset_action")
        exponentialDecaySlider = dpg.add_slider_float(label = "exponential decay", default_value = exponentialDecay, min_value = 0, max_value = 1, tag = "exponential_decay_action")
        pumpThresholdSlider = dpg.add_slider_int(label = "pump threshold", default_value = pumpThreshold, min_value = 2, max_value = 100, tag = "pump_threshold_action")
        pumpHeightThresholdSlider = dpg.add_slider_int(label = "pump height cutoff", default_value = pumpHeightThreshold, min_value = 0, max_value = 100, tag = "pump_height_threshold_action")
        filterOrderSlider = dpg.add_slider_int(label = "filter order", default_value = useOrder, min_value = 1, max_value = 3, tag = "filter_order_action")
        referenceBarSlider = dpg.add_slider_int(label = "reference bar", default_value = referenceBar, min_value = 0, max_value = bars - 1, tag = "reference_bar_action")
        dpg.add_text("")
        ledVignettePowerSlider = dpg.add_slider_float(label = "led vignette power", default_value = ledVignettePower, min_value = 0, max_value = 4, tag = "led_vignette_power_action")
        ledVignetteMultiplierSlider = dpg.add_slider_float(label = "led vignette multiplier", default_value = ledVignetteMultiplier, min_value = 0, max_value = 4, tag = "led_vignette_multiplier_action")

    anchor = 53

    with dpg.child_window(tag = "main_parent_group_2", pos = (515 + 35, anchor), border = False, resizable_x = False, width = 515):
        dpg.add_text("layer settings")
        layerPowerSlider = dpg.add_slider_float(label = "layer power", default_value = layerPower, min_value = 0.1, max_value = 2, tag = "layer_power_action")
        layerMultiplierSlider = dpg.add_slider_float(label = "layer multiplier", default_value = layerMultiplier, min_value = 0.2, max_value = 10, tag = "layer_multiplier_action")
        layerOffsetSlider = dpg.add_slider_float(label = "layer offset", default_value = layerOffset, min_value = -50, max_value = 50, tag = "layer_offset_action")
        layerHueSlider = dpg.add_slider_float(label = "layer hue", default_value = layerHue, min_value = 0, max_value = 1, tag = "layer_hue_action")
        layerSatSlider = dpg.add_slider_float(label = "layer saturation", default_value = layerSat, min_value = 0, max_value = 1, tag = "layer_sat_action")
        layerOpacitySlider = dpg.add_slider_float(label = "layer opacity", default_value = layerOpacity, min_value = 0, max_value = 1, tag = "layer_opacity_action")
        dpg.add_text("")
        dpg.add_text(f"layer mode: {layerModeDict[layerMode]}", tag = "layer_mode_text")
        layerModeSlider = dpg.add_slider_int(label = "layer mode", default_value = layerMode, min_value = 0, max_value = 2, tag = "layer_mode_action")
        layerHueSpreadSlider = dpg.add_slider_float(label = "layer hue spread", default_value = layerHueSpread, min_value = -0.1, max_value = 0.1, tag = "layer_hue_spread_action")
        layer_sideOrientationSlider = dpg.add_slider_int(label = "layer orientation", default_value = layer_sideOrientation, min_value = 0, max_value = 2, tag = "layer_side_orientation_action")
        layer_pulseThresholdSlider = dpg.add_slider_int(label = "pulse threshold", default_value = layer_pulseThreshold, min_value = 0, max_value = 125, tag = "layer_pulse_threshold_action")
        layer_pulseDecayRateSlider = dpg.add_slider_float(label = "pulse decay rate", default_value = layer_pulseDecayRate, min_value = 0, max_value = 1, tag = "layer_pulse_decay_rate_action")
        dpg.add_text("")
        dpg.add_text("base settings")
        baseMultiplierSlider = dpg.add_slider_float(label = "base multiplier", default_value = baseMultiplier, min_value = 0, max_value = 1, tag = "base_multiplier_action")
        baseHueSlider = dpg.add_slider_float(label = "base hue", default_value = baseHue, min_value = 0, max_value = 1, tag = "base_hue_action")
        baseSatSlider = dpg.add_slider_float(label = "base saturation", default_value = baseSat, min_value = 0, max_value = 1, tag = "base_sat_action")
        dpg.add_text("")
        dpg.add_text(f"base mode: {baseModeDict[baseMode]}", tag = "base_mode_text")
        baseModeSlider = dpg.add_slider_int(label = "base mode", default_value = baseMode, min_value = 0, max_value = 2, tag = "base_mode_action")
        base_rainbowSpeedSlider = dpg.add_slider_float(label = "gradient speed", default_value = base_rainbowSpeed, min_value = 0, max_value = 10, tag = "base_rainbow_speed_action")
        base_rainbowScaleSlider = dpg.add_slider_float(label = "gradient scale", default_value = base_rainbowScale, min_value = 0, max_value = 2, tag = "base_rainbow_scale_action")
        base_rainbowStartSlider = dpg.add_slider_float(label = "gradient start", default_value = base_rainbowStart, min_value = 0, max_value = 1, tag = "base_rainbow_start_action")
        base_rainbowEndSlider = dpg.add_slider_float(label = "gradient end", default_value = base_rainbowEnd, min_value = 0, max_value = 1, tag = "base_rainbow_end_action")
        base_breathingSpeedSlider = dpg.add_slider_float(label = "breathing speed", default_value = base_breathingSpeed, min_value = 0, max_value = 10, tag = "base_breathing_speed_action")
        dpg.add_text("")
        dpg.add_text("presets")
        presetCombo = dpg.add_combo(label = "select preset", items = ["Purple Rainbow", "Purple Blue Rainbow", "Neon Rainbow", "Redshift Rainbow", "Torch", "Warm", "Calm"], default_value = "Purple Rainbow", tag = "preset_action", callback = LED_update_base_preset)
        dpg.add_text("")
        colorContributionThresholdSlider = dpg.add_slider_float(label = "color contribution threshold", default_value = colorContributionThreshold, min_value = 0, max_value = 1, tag = "color_contribution_threshold_action")
        colorContributionRatioSlider = dpg.add_slider_float(label = "color contribution ratio", default_value = colorContributionRatio, min_value = 0, max_value = 1, tag = "color_contribution_ratio_action")
        colorValueThresholdSlider = dpg.add_slider_int(label = "color value threshold", default_value = colorValueThreshold, min_value = 0, max_value = 255, tag = "color_value_threshold_action")
        colorValueCutSlider = dpg.add_slider_float(label = "color value cut", default_value = colorValueCut, min_value = 0, max_value = 1, tag = "color_value_cut_action")
        colorTransitionSpeedSlider = dpg.add_slider_float(label = "color transition speed", default_value = colorTransitionSpeed, min_value = 0, max_value = 4, tag = "color_transition_speed_action")
        dpg.add_text("")
        colorCalibrationRSlider = dpg.add_slider_float(label = "color calibration r", default_value = colorCalibrationR, min_value = 0, max_value = 1, tag = "color_calibration_r_action")
        colorCalibrationGSlider = dpg.add_slider_float(label = "color calibration g", default_value = colorCalibrationG, min_value = 0, max_value = 1, tag = "color_calibration_g_action")
        colorCalibrationBSlider = dpg.add_slider_float(label = "color calibration b", default_value = colorCalibrationB, min_value = 0, max_value = 1, tag = "color_calibration_b_action")
        
    # base individual
    # anchor = dpg.get_item_pos("base_group")[1] + dpg.get_item_rect_size("base_group")[1]
    


with dpg.window(label = "Raw", tag = "raw_window", width = 515, height = 715):
    for i in range(bars):
        dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
        dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - rawHeight[i]), (10 + 15 * i, baseYPos - rawHeight[i]), color = (220, 220, 220), tag = f"dynamic_box_raw{i}")
    dpg.draw_text((30, baseYPos + 30), f"current device id: {DEVICE_ID}", color = (250, 250, 250, 255), size = 15)
    dpg.draw_text((30, baseYPos + 50), "frametime: 0ms", color = (250, 250, 250, 255), size = 15, tag = "frametime")
    fpsSlider = dpg.add_slider_int(label = "target fps", pos = (30, baseYPos + 105), default_value = fps, min_value = 5, max_value = 240, tag = "fps_action")
    with dpg.plot(label = "waveform", height = 200, width = 455, pos = (30, baseYPos + 130)):
        dpg.add_plot_legend()
        dpg.add_plot_axis(dpg.mvXAxis, label = "sample")
        dpg.add_plot_axis(dpg.mvYAxis, label = "", tag = "y_axis")
        dpg.set_axis_limits("y_axis", -32000, 32000)
        dpg.add_line_series(shortWaveformx, shortWaveform, label = "left channel", tag = "waveform_series", parent = "y_axis")

ui_update_menu_items()

# handler

handles = {"exit": exit_program, 
           "bake_mode": update_bake_mode,
           "decay_speed": update_properties,
           "power_multiplier": update_properties,
           "power_offset": update_properties,
           "exponential_decay": update_properties,
           "pump_threshold": update_properties,
           "pump_height_threshold": update_properties,
           "filter_order": update_properties,
           "reference_bar": update_properties,
           "layer_mode": ui_update_menu_items,
           "layer_power": update_properties,
           "layer_multiplier": update_properties,
           "layer_offset": update_properties,
           "layer_hue": update_properties,
           "layer_sat": update_properties,
           "layer_opacity": update_properties,
           "layer_side_orientation": update_properties,
           "layer_hue_spread": update_properties,
           "layer_pulse_threshold": update_properties,
           "layer_pulse_decay_rate": update_properties,
           "base_mode": ui_update_menu_items,
           "base_hue": update_properties,
           "base_rainbow_speed": update_properties,
           "base_multiplier": update_properties,
           "base_sat": update_properties,
           "base_rainbow_scale": update_properties,
           "base_breathing_speed": update_properties,
           "fps": update_properties,
           "base_rainbow_start": update_properties,
           "base_rainbow_end": update_properties,
           "led_vignette_power": update_properties,
           "led_vignette_multiplier": update_properties,
           "color_contribution_threshold": update_properties,
           "color_contribution_ratio": update_properties,
           "color_value_threshold": update_properties,
           "color_value_cut": update_properties,
           "color_transition_speed": update_properties,
           "color_calibration_r": update_properties,
           "color_calibration_g": update_properties,
           "color_calibration_b": update_properties,
           }

for handle in handles:
    with dpg.item_handler_registry(tag = f"{handle}_handler") as handler:
        dpg.add_item_active_handler(callback = handles[handle])
    dpg.bind_item_handler_registry(f"{handle}_action", f"{handle}_handler")

# draw ui
dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1545 + 15, height = 715 + 35)
dpg.set_viewport_pos((360, 165))

dpg.setup_dearpygui()
dpg.show_viewport()

# frame update
while dpg.is_dearpygui_running():
    startTime = time.time()
    frame()
    dpg.render_dearpygui_frame()
    endTime = time.time()
    frametime = (endTime - startTime)
    time.sleep(max((1/fps) - (frametime), 0))
    dpg.configure_item("frametime", text = f"process frametime: {frametime * 1000:.0f}ms")