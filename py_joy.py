import vgamepad as vg
import time
#import serial

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

def compare(value,lastvalue):
    limit_val = 1
    if abs(value - lastvalue) < 0.01:
        val = lastvalue
    else:
        val = value
    return val



#x_sens = 2
#y_sens = 2
#z_sens = 2
#arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)
gamepad = vg.VX360Gamepad()

#time.sleep(10)
#frontlader arm heben x = -1
#gamepad.right_joystick_float(x_value_float=0, y_value_float=0)
#gamepad.update()
#time.sleep(1)
#gamepad.right_joystick_float(x_value_float=-1, y_value_float=0)
#gamepad.update()
#time.sleep(1)
#gamepad.right_joystick_float(x_value_float=0, y_value_float=0)
#gamepad.update()
#time.sleep(1)

#frontlader werkzeug öffnen y = 1
#gamepad.right_joystick_float(x_value_float=0, y_value_float=0)
#gamepad.update()
#time.sleep(1)
#gamepad.right_joystick_float(x_value_float=0, y_value_float=1)
#gamepad.update()
#time.sleep(1)
#gamepad.right_joystick_float(x_value_float=0, y_value_float=0)
#gamepad.update()
#time.sleep(1)


x = input("Warte auf Eingabe")
print(x)
time.sleep(3)
print("starting")

# while True:

    # gamepad.right_joystick_float(x_value_float=0.1, y_value_float=0.1)
    # gamepad.update()
    # time.sleep(1)
    # gamepad.right_joystick_float(x_value_float=0.7, y_value_float=0.7)
    # gamepad.update()
    # time.sleep(1)




# press buttons and things
#gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
#gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
#gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN)
#gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
#gamepad.left_trigger_float(value_float=0.5)
#gamepad.right_trigger_float(value_float=0.5)
#gamepad.left_joystick_float(x_value_float=0.0, y_value_float=0.2)
first = True
while True:
    for i in range(-100,100):
        x = i * 0.1
        y = i * 0.1
        if first:
            x = 100 - x
            y = 100 - y
        
        
        #x -1 1 links nach rechts
        #y -1 1 unten nach oben
        #gamepad.right_joystick_float(x_value_float=x, y_value_float=y)
        gamepad.left_joystick_float(x_value_float=x, y_value_float=0)
        gamepad.update()
        print(x,y)
        time.sleep(0.1)
    first = False
# release buttons and things
#gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
#gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT)
#gamepad.right_trigger_float(value_float=0.0)
#gamepad.right_joystick_float(x_value_float=0.0, y_value_float=0.0)

# reset gamepad to default state
gamepad.reset()
