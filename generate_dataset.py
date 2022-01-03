import pickle
import struct

import time
import tminterface.client
from tminterface.interface import TMInterface
import os
import sys
import ctypes
import numpy as np
from replay_input import get_inputs_from_replay, event_to_analog_value, get_positions_from_replay
from utils import ScreenshotHelper
import win32ui
import cv2

window_name = "TmForever"

if len(sys.argv) != 3:
    raise RuntimeError("usage: tminterfacetest <TmForever.exe> <folder>")

wnd = win32ui.FindWindow(None, window_name)
if wnd is None:
    raise RuntimeError("trackmania not running...")

screenshot_helper = ScreenshotHelper(wnd.GetSafeHwnd())

class MyClient(tminterface.client.Client):
    def __init__(self, tm_executable, track_path, replay_path):
        super().__init__()
        os.system(f'\"\"{tm_executable}\" /useexedir /singleinst /file=\"{track_path}\"\"')
        time.sleep(5)
        self.track_path = track_path
        self.out = cv2.VideoWriter(f'{track_path}.mp4',cv2.VideoWriter_fourcc(*'x264'), 100, (640,480))
        self.inputs = get_inputs_from_replay(replay_path)
        self.states = []

    def on_registered(self, iface: TMInterface):
        print("on_registered")
        # iface.execute_command("set speed 0.1")
        iface.execute_command("set countdown_speed 5")
        iface.execute_command("set skip_map_load_screens true")
        pass

    def on_deregistered(self, iface):
        print("on_deregistered")
        pass

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time < 0:
            return

        img = screenshot_helper.screenshot()
        img = cv2.flip(img, 0)
        self.out.write(img)

        self.states.append(iface.get_simulation_state())

        i = 0
        while i < len(self.inputs):
            (time, event) = self.inputs[i]
            if time == _time:
                print(f"event={event.event_name} enabled={event.enabled}")
                if event.event_name == 'Accelerate' or event.event_name == 'AccelerateReal':
                    iface.set_input_state(accelerate=event.enabled)
                elif event.event_name == 'SteerLeft':
                    iface.set_input_state(left=event.enabled)
                elif event.event_name == 'SteerRight':
                    iface.set_input_state(right=event.enabled)
                elif event.event_name == 'Brake' or event.event_name == 'BrakeReal':
                    iface.set_input_state(brake=event.enabled)
                else:
                    raise RuntimeError(f'event type {event.event_name} not allowed')
            elif time > _time:
                break

            i += 1

        # remove old stuff from list
        self.inputs = self.inputs[i:]

    def on_simulation_begin(self, iface):
        print("on_simulation_begin")
        pass

    def on_simulation_step(self, iface, _time: int):
        print("on_simulation_step")
        pass

    def on_simulation_end(self, iface, result: int):
        print("on_simulation_end")
        pass

    def on_checkpoint_count_changed(self, iface : TMInterface, current: int, target: int):
        print(f"on_checkpoint_count_changed {current} {target}")
        if current == target:
            self.out = None
            pickle.dump(self.states, open(self.track_path + ".states.bin", 'wb'))
            iface.close()
        pass

    def on_laps_count_changed(self, iface, current: int):
        print(f"on_laps_count_changed {current}")
        pass

    def on_custom_command(self, iface, time_from: int, time_to: int, command: str, args: list):
        print("on_custom_command")
        pass


if os.path.isdir(sys.argv[2]):
    folder = os.path.abspath(sys.argv[2])
    for path in os.listdir(folder):
        if (path.endswith(".Challenge.Gbx")):
            print(path)
            tminterface.client.run_client(MyClient(sys.argv[1], folder + "/" + path, folder + "/" + path.replace(".Challenge.", ".Replay.")), "TMInterface0")
else:
    file = os.path.abspath(sys.argv[2])
    tminterface.client.run_client(MyClient(sys.argv[1], file, file.replace(".Challenge.", ".Replay.")), "TMInterface0")
