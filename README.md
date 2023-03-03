# Autosteer
The Autosteer project is a python script designed to automatically steer a vehicle in farming simulator.

<img src="https://github.com/mgoetze06/autosteer/blob/main/vlcsnap-2023-01-13-23h08m44s198.png?raw=true" width="100%">

## Grabbing Window as Screenshot
> import win32gui <br/>
> https://pypi.org/project/win32gui/

## Filtering Road Lines
The script uses scikit-image library to parse image data. The Filter threshold_yen is used to transform image to binary image.
> contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  <br/>
> image = cv2.drawContours(org_img, contours[0], -1, (0,255,0), 3) <br/>

All contours that are not in the central area (vehicle) are used to generate line with linear regression
>  coeffs = np.polyfit(x_np, y_arr, degree)

## Calculate Difference
Afterwards the horizontal distance between the middle point of the screen and the line is calculated with the function
> calculate_difference_line

## PID Control
The difference is then used as the input of a simple PID controller which then outputs the steering in x-direction to a virtual gamepad
> from simple_pid import PID <br/>
> https://pypi.org/project/simple-pid/ <br/>
> import vgamepad as vg <br/>
> https://pypi.org/project/vgamepad/ <br/>

