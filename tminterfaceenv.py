import pickle
import signal
import struct
import subprocess
import threading

import cv2
import math
import time
import tminterface.client
import torch
from tminterface.interface import TMInterface
import os
import sys
import ctypes
import numpy as np

from model import sim_state_to_state_input
from replay_input import get_inputs_from_replay, event_to_analog_value, get_positions_from_replay
import gym
from utils import ScreenshotHelper
import win32ui
import win32gui
import win32process
import win32api
import win32con


class MyClient(tminterface.client.Client):
    def __init__(self):
        self.task = None
        self.response = None
        self.current_state = None
        self.start_state = None
        self.num_missed_steps = 0
        self.done = False

    def on_registered(self, iface: TMInterface):
        print("on_registered")
        # iface.execute_command("set speed 0.1")
        iface.execute_command("toggle_console")
        iface.execute_command("set autologin 1")
        iface.execute_command("set countdown_speed 5")
        iface.execute_command("set skip_map_load_screens true")
        iface.execute_command("press right")
        pass

    def on_deregistered(self, iface):
        print("on_deregistered")
        pass

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time < 0:
            return

        state = iface.get_simulation_state()
        if self.start_state is None:
            self.start_state = state
        if self.current_state is None:
            self.current_state = state

        if self.task is not None:
            # print(f"missed steps: {self.num_missed_steps}")
            self.num_missed_steps = 0
            if self.task == "reset":
                iface.rewind_to_state(self.start_state)
                self.current_state = self.start_state
            else:
                # rewind and change inputs
                iface.rewind_to_state(self.current_state)
                iface.set_input_state(accelerate=self.task[0], brake=self.task[1], left=self.task[2],
                                      right=self.task[3])
                self.current_state = None

            self.task = None
            self.response = state
        else:
            self.num_missed_steps += 1
            # just rewind to stay on the same state
            iface.rewind_to_state(self.current_state)

    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        print(f"on_checkpoint_count_changed {current} {target}")
        self.done = current == target
        pass


class TmInterfaceEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 255, [3, 480, 640], dtype=np.uint8)
    action_space = gym.spaces.Box(0, 1, [4, 1], dtype=np.float32)

    FREE_INSTANCES = [f"TMInterface{i}" for i in range(16)]
    instances = []

    def __init__(self, tm_executable, track, with_state_output):
        super(TmInterfaceEnv, self).__init__()

        self.with_state_output = with_state_output

        print(f"TmInterfaceEnv __init__('{tm_executable}','{track}')")
        # script_dir = os.path.dirname(os.path.realpath(__file__))
        subprocess.run([tm_executable], cwd=os.path.dirname(tm_executable)).check_returncode()
        # wait 3 seconds for the window to show up
        time.sleep(3)

        windows = []
        win32gui.EnumWindows(
            lambda hwnd, extra: windows.append(hwnd) if win32gui.GetWindowText(hwnd) == "TmForever" else None, None)
        if len(windows) == 0:
            raise RuntimeError("trackmania not running...")
        if len(windows) != 1:
            raise RuntimeError("multiple windows found...")

        # try to find a free instance:
        if len(TmInterfaceEnv.FREE_INSTANCES) == 0:
            raise RuntimeError("too many TMInterface instances")

        self.client = MyClient()

        for instance in TmInterfaceEnv.FREE_INSTANCES:
            print("trying to connect to interface", instance)
            self.iface = TMInterface(instance)
            self.iface.register(self.client)
            try:
                self._wait_for_register(1)
                print("success")
                break
            except RuntimeError:
                print("failed")
                self.iface.close()
                self.iface = None
                continue

        def handler(signum, frame):
            for i in TmInterfaceEnv.instances:
                i.close()

            exit()

        signal.signal(signal.SIGBREAK, handler)
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

        if self.iface is None:
            raise RuntimeError("could not connect to interface")

        TmInterfaceEnv.FREE_INSTANCES.remove(self.iface.server_name)
        TmInterfaceEnv.instances.append(self)
        tm_window = windows[0]
        _, self.tm_process = win32process.GetWindowThreadProcessId(tm_window)

        win32gui.SetWindowText(tm_window, self.iface.server_name)
        self.screenshot_helper = ScreenshotHelper(tm_window)

        # send data copy message to window with the file name
        class CopyDataStruct(ctypes.Structure):
            _fields_ = [("dwData", ctypes.c_size_t), ("cbData", ctypes.c_long), ("lpData", ctypes.c_void_p)]

        str_buff = ctypes.create_unicode_buffer(track)
        struct = CopyDataStruct()
        struct.dwData = 0x46494C45
        struct.cbData = ctypes.sizeof(str_buff)
        struct.lpData = ctypes.addressof(str_buff)
        result = win32gui.SendMessage(tm_window, 0x4A, 0, ctypes.addressof(struct))
        if result == 0:
            raise RuntimeError(f"TmForever could not load track '{track}'")

        # reset with extra timeout
        self.client.task = "reset"
        self._wait_for_reponse(200)

        self.replay_states = pickle.load(open(track + ".states.bin", "rb"))

    def get_screen(self):
        # transpose to create a 3xhxw array from a hxwx3 array
        # double() to convert to float64
        # unspeeze to add the batch dimension
        img = self.screenshot_helper.screenshot()
        img = cv2.flip(img, 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.transpose(img, axes=(2,0,1))

    def reset(self):
        self.client.task = "reset"
        self._wait_for_reponse(1)
        output_state = self.get_screen()
        if self.with_state_output:
            output_state = (output_state, sim_state_to_state_input(0, self.replay_states))
        return output_state

    def step(self, input):
        assert input.shape == (1,4)
        self.client.task = [input[0,0] > 0.5, input[0,1] > 0.5, input[0,2] > 0.5, input[0,3] > 0.5]
        state = self._wait_for_reponse(1)
        dist = np.linalg.norm(np.array(state.position) - np.array(
            self.replay_states[min(state.race_time // 10, len(self.replay_states)-1)].position))
        reward = math.exp(-dist)
        #print(f"env.step({input}) -> reward={reward} dist={dist}")
        done = state.race_time > 10000 or self.client.done
        output_state = self.get_screen()
        if self.with_state_output:
            output_state = (output_state, sim_state_to_state_input(state.race_time // 10, self.replay_states))
        return output_state, reward, done, dict()

    def _wait_for_reponse(self, timeout_s):
        reponse = None
        wait_start = time.monotonic()
        while True:
            if self.client.response is not None:
                reponse = self.client.response
                self.client.response = None
                return reponse

            if time.monotonic() - wait_start > timeout_s:
                raise RuntimeError("_wait_for_reponse timeout")
            time.sleep(0)

    def _wait_for_register(self, timeout_s):
        wait_start = time.monotonic()
        while True:
            if self.iface.registered:
                return

            if time.monotonic() - wait_start > timeout_s:
                raise RuntimeError("_wait_for_register timeout")

            time.sleep(0.001)

    def close(self):
        self.iface.close()
        handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, 0, self.tm_process)
        win32api.TerminateProcess(handle, 0)
        win32api.CloseHandle(handle)
