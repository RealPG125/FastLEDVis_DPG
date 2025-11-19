g.child_window(label = "processed", width = 515, height = 370):
        for i in range(bars):
            dpg.draw_text((10 + 15 * i, baseYPos + 10), str(height[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text{i}")
            dpg.draw_quad((10 + 15 * i, baseYPos), (20 + 15 * i, baseYPos), (20 + 15 * i, baseYPos - height[i]), (10 + 15 * i, baseYPos - height[i]), fill = (200, 255, 255), tag = f"dynamic_box{i}")
        dpg.draw_text((10, baseYPos + 25), f"window size: {dpg.get_item_width('main_window')}x{dpg.get_item_height('main_window')}", size = 15, tag = "window_size_text")
        dpg.add_button(pos = (30, baseYPos + 80), label = "exit", tag = "exit_button")
        decaySpeedSlider = dpg.add_slider_float(label = "decay speed", pos = (30, baseYPos + 120), default_value = 1, min_value = 0.25, max_value = 4)
        powerMultiplierSlider = dpg.add_slider_float(label = "power multiplier", pos = (30, baseYPos + 140), default_value = 1, min_value = 0.25, max_value = 4)
        filterOrderSlider = dpg.add_slider_int(label = "filter order", pos = (30, baseYPos + 180), default_value = 2, min_value = 1, max_value = 3)
    with dpg.child_window(label = "raw", width = 515, height = 370):
        for i in range(bars):
            dpg.draw_text((10 + 15 * i, baseYPos + 10), str(rawHeight[i]), color = (250, 250, 250, 255), size = 15, tag = f"height_text_raw{i}")
            dpg.dr