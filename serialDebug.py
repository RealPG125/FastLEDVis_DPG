import dearpygui.dearpygui as dpg

dpg.create_context()

with dpg.window(label = "Main", tag = "main_window", width = 1030, height = 600):
    decaySpeedSlider = dpg.add_slider_float(label = "decay speed", default_value = 1, min_value = 0, max_value = 10, tag = "decay_speed_action")
    dpg.set_value("decay_speed_action", 5.0)
dpg.create_viewport(title = 'Visualizer UI - DPG', width = 1545 + 15, height = 600 + 35)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

print(dpg.get_item_configuration("decay_speed_action"))