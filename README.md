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



# Autosteer swath detection
Automatically follow a swath by steering a vehicle in farming simulator.

<img src="https://github.com/mgoetze06/autosteer/blob/main/schwaderkennung/schwaderkennung.png?raw=true" width="100%">

## Get swath position
Rectangle ROI in center of screenshot, height offset to match the vehicle engine hood.
Use maximum and minimum gradient on grayscale image to detect swath after blurring the image.
>blur_gray = cv2.GaussianBlur(img_gray_res,(kernel_size, kernel_size),0)<br/>
>up = np.argmax(np.gradient(avg))<br/>
>down = np.argmin(np.gradient(avg))<br/>

The following graph shows the intensity change of the pixels in a given line. Light blue line is the image center. Red line is the swath center calculated from both blue lines.
<img src="https://github.com/mgoetze06/autosteer/blob/main/schwaderkennung/schwad_lines.png?raw=true" width="100%">

Calculate distance of swath to the center of the image.

<img src="https://github.com/mgoetze06/autosteer/blob/main/schwaderkennung/schwad.png?raw=true" width="100%">

## PID Control

Use PID controller to control the distance between swath and center of image to zero. 

<img src="https://github.com/mgoetze06/autosteer/blob/main/schwaderkennung/schwadregelung.png?raw=true" width="100%">

