#img = io.imread('./images/img5.jpg')
from autosteerfunctions import calculate_difference_line
from simple_pid import PID
import cv2
import numpy as np
import win32gui
from PIL import ImageGrab
import time
import vgamepad as vg

time.sleep(3)
gamepad = vg.VX360Gamepad()
windows_list = []
toplist = []
def enum_win(hwnd, result):
    win_text = win32gui.GetWindowText(hwnd)
    windows_list.append((hwnd, win_text))
win32gui.EnumWindows(enum_win, toplist)

def limit(value):
    limit_val_up = 1
    limit_val_down = -1
    if value >= limit_val_up:
        val = limit_val_up
    else:
        if value <= limit_val_down:
            val = limit_val_down
        else:
            val = value 
    return val

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
pid = PID(40, 1.2, 0.2, setpoint=93)
setpoint = 93
use_pid = True
while True:
    screenshot = ImageGrab.grab(position)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    #screenshot = cv2.resize(screenshot,(960,540))
    diff_line,image = calculate_difference_line(screenshot,True,nbins=256)
    print(diff_line)
    if use_pid:
        if diff_line != 0:
            pid_out = pid(diff_line)
            new_gamepad_x = limit(pid_out/screenshot.shape[1])
            print("new_gamepad_x",new_gamepad_x)
        else:
            new_gamepad_x = 0
    else:
        if diff_line != 0:
            new_gamepad_x = limit((setpoint/diff_line) - 1)
        else:
            new_gamepad_x = 0
    gamepad.left_joystick_float(x_value_float=new_gamepad_x, y_value_float=0)
    gamepad.update()
    screenshot = cv2.resize(image,(960,540))
    window_name = "Screen"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, -1200,30)
    cv2.imshow(window_name, screenshot)
    cv2.waitKey(1)
cv2.destroyAllWindows()
    


#diff_line,image = calculate_difference_line(img,False)
#if image is not None:
#    plt.imshow(image)
#print(diff_line)