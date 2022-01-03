import ctypes
import struct
import numpy as np
import time

import win32ui
import win32gui
import win32con

#https://stackoverflow.com/questions/5750887/python-use-windows-api-to-render-text-using-a-ttf-font
def native_bmp_to_np(hdc, bitmap_handle, width, height):
    bmpheader = struct.pack("LHHHH", struct.calcsize("LHHHH"),
                            width, height, 1, 24) #w,h, planes=1, bitcount)
    c_bmpheader = ctypes.c_buffer(bmpheader)

    width_padded = ((width*3 + 3) & -4)
    #3 bytes per pixel, pad lines to 4 bytes
    c_bits = ctypes.c_buffer((height * width_padded))

    res = ctypes.windll.gdi32.GetDIBits(
        hdc, bitmap_handle, 0, height,
        c_bits, c_bmpheader,
        win32con.DIB_RGB_COLORS)
    if not res:
        raise IOError("native_bmp_to_pil failed: GetDIBits")

    #return np.lib.stride_tricks.as_strided(np.frombuffer(c_bits, np.uint8), shape=(height,width,3), strides=(width_padded, 3, 1))
    return np.reshape(np.reshape(np.frombuffer(c_bits, np.uint8), (height, width_padded))[:, 0:width*3], (height, width, 3))

class ScreenshotHelper:
    def __init__(self, hwnd):
        self.hwnd = hwnd

        l, t, r, b = win32gui.GetClientRect(self.hwnd)
        l, t = win32gui.ClientToScreen(self.hwnd, (l, t))
        r, b = win32gui.ClientToScreen(self.hwnd, (r, b))
        wl, wt, wr, wb = win32gui.GetWindowRect(self.hwnd)
        self.client_resolution = (r - l, b - t)
        self.client_offset = (l - wl, t - wt)
        self.hDC = win32gui.GetWindowDC(self.hwnd)
        self.myDC = win32ui.CreateDCFromHandle(self.hDC)
        self.newDC = self.myDC.CreateCompatibleDC()

        self.myBitMap = win32ui.CreateBitmap()
        self.myBitMap.CreateCompatibleBitmap(self.myDC, self.client_resolution[0], self.client_resolution[1])
        self.newDC.SelectObject(self.myBitMap)

    def screenshot(self):
        start = time.perf_counter()
        self.newDC.BitBlt((0, 0), self.client_resolution, self.myDC, self.client_offset, win32con.SRCCOPY)
        # self.myBitMap.SaveBitmapFile(self.newDC,'tmp.bmp')
        im = native_bmp_to_np(self.newDC.GetSafeHdc(), self.myBitMap.GetHandle(), self.client_resolution[0], self.client_resolution[1])
        end = time.perf_counter()
        #print(f"screenshot took {(end - start) * 1000}ms")
        return im