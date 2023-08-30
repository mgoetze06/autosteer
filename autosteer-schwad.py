#img = io.imread('./images/img5.jpg')
from autosteerfunctions import calculate_difference_line,rectangle_roi_center,get_schwad_distance_from_center
from simple_pid import PID
import cv2
import numpy as np
import win32gui
from PIL import ImageGrab
import time
import matplotlib.pyplot as plt 
import vgamepad as vg
from win32api import GetKeyState
import win32api
import time

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
pid = PID(1, 0.1, 0.01, setpoint=0)
#pid = PID(5, 1, 0.01, setpoint=93)
#setpoint = 93
#use_pid = True
gamepad_x = 0
freigabe = False
while True:
    if win32api.GetAsyncKeyState(13):
        print("enter pressed")
        freigabe = not freigabe
        gamepad.left_joystick_float(x_value_float=0, y_value_float=0)
        gamepad.update()
    time.sleep(0.1)
    screenshot = ImageGrab.grab(position)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    #screenshot = cv2.resize(screenshot,(960,540))    
    roi = rectangle_roi_center(screenshot,0.06666,0.08666,-150,0)
    schwad_fehler,roi = get_schwad_distance_from_center(roi,50,False)
    if freigabe:
        print("schwadfehler: ",schwad_fehler)
        pid_out = pid(schwad_fehler)
        print("PID_Out: ",pid_out)
        new_gamepad_x = limit(round(pid_out/300,2))

        if abs(new_gamepad_x - gamepad_x)>1:
            print("Fehler aufgetreten")
            gamepad.left_joystick_float(x_value_float=0, y_value_float=0)
            gamepad.update()
        else:

            print("newgamepadx: ",new_gamepad_x)

            gamepad.left_joystick_float(x_value_float=new_gamepad_x, y_value_float=0)
            gamepad.update()
        gamepad_x = new_gamepad_x
        #screenshot = cv2.resize(roi,(960,540))
    window_name = "Screen"
    cv2.namedWindow(window_name)
    cv2.moveWindow(window_name, -1200,30)
    cv2.imshow(window_name, roi)
    cv2.waitKey(1)
cv2.destroyAllWindows()
    


#diff_line,image = calculate_difference_line(img,False)
#if image is not None:
#    plt.imshow(image)
#print(diff_line)