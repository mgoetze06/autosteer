import cv2
import numpy as np
import win32gui
from PIL import ImageGrab
import time


time.sleep(3)

windows_list = []
toplist = []
def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))
win32gui.EnumWindows(enum_win, toplist)

# Game handle
game_hwnd = 0
for (hwnd, win_text) in windows_list:
    #print(hwnd)
    print(win_text)
    if win_text == "Farming Simulator 22":
        print("found game")
        game_hwnd = hwnd
        break
        
position = win32gui.GetWindowRect(game_hwnd)
while True:
    screenshot = ImageGrab.grab(position)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = cv2.resize(screenshot,(960,540))


    window_name = "Screen"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, -1200,30)
    cv2.imshow(window_name, screenshot)
    cv2.waitKey(1)
cv2.destroyAllWindows()