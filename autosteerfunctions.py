from skimage.filters import try_all_threshold, threshold_yen
from skimage import color,io,exposure
from skimage.morphology import erosion, dilation, opening, closing, diamond
from sklearn.linear_model import LinearRegression
from scipy import stats
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import sklearn
import glob
import math

def calculate_difference_line(img,show=False,nbins=128,displayCalculations=False):
    diff_line = None
    def not_in__middle(x,y,dist,image,draw=False):
        x_mid = image.shape[1]//2
        y_mid = image.shape[0]//2

        #return true if pixel is in desired area out of middle area
        #true if pixel should be used for regression
        #pixel is in area:
        middle_x = (x < x_mid - dist or x > x_mid + dist)
        middle_y = (y < y_mid - dist or y > y_mid + dist)
        left_and_right_edge = (x > 300 and x < 1200)
        
        #return (x < x_mid - dist or x > x_mid + dist) and (y < y_mid - dist or y > y_mid + dist)
        if draw:
            image = cv2.rectangle(image, (x_mid - dist,y_mid -dist), (x_mid + dist,y_mid + dist), (0,0,255), 3)
            #manual rectangle
            #image = cv2.line(image,(x_mid - dist,y_mid -dist),(x_mid - dist,y_mid +dist),(0,0,255),1)
            #image = cv2.line(image,(x_mid - dist,y_mid -dist),(x_mid + dist,y_mid -dist),(0,0,255),1)
            #image = cv2.line(image,(x_mid + dist,y_mid -dist),(x_mid + dist,y_mid +dist),(0,0,255),1)
            #image = cv2.line(image,(x_mid + dist,y_mid +dist),(x_mid + dist,y_mid -dist),(0,0,255),1)
            image = cv2.line(image,(300,0),(300,2*y_mid-1),(0,0,255),1)
            image = cv2.line(image,(1200,0),(1200,2*y_mid-1),(0,0,255),1)
        
        
        
        return (middle_x and middle_y and left_and_right_edge),image
    org_img = img
    image = np.copy(img)
    image = color.rgb2gray(image)
    thresh = threshold_yen(image, nbins=nbins)
    if displayCalculations:
        print(thresh)
    image_bin = image > thresh
    elements = [np.ones((1,2), dtype=int),np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])]
    for element in elements:
        if displayCalculations:
            print(element)
        image_bin = erosion(image_bin, footprint=element)
    if displayCalculations:
        print("Maximalwert des Binärbildes: ",np.max(image_bin))
        print("Datentyp des Binärbildes: ",image_bin.dtype)
    image = exposure.rescale_intensity(image_bin,in_range=(0,1),out_range='uint8')
    contours = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if show:
        image = cv2.drawContours(org_img, contours[0], -1, (0,255,0), 3)
    x_arr = []
    y_arr = []
    if len(contours[0]):
        for c in contours[0]:
            if len(c) > 12:
                (x,y) = (c[0][0][0],c[0][0][1])
                cond,image = not_in__middle(x,y,70,image,draw=True)
                if x > 500 and x < 1500 and cond:
                    cv2.circle(image, (x,y),10,(0,0,0))
                    for element in c:
                        x_arr.append([x])
                        y_arr.append([y])
                
        x_arr = np.array([x_arr]).reshape((-1, 1))#.ravel()
        y_arr = np.array([y_arr]).reshape((1,-1)).ravel()
        if displayCalculations:
            print("Konturpunkte x Shape: ",x_arr.shape)
            print("Konturpunkte y Shape: ",y_arr.shape)
        if len(x_arr) > 2:
            #linear regression with sklearn
            #model = LinearRegression()
            #pred = model.fit(x_arr,y_arr)
            x_1d = np.arange(600,1100) #hier muss theoertisch nur in der Mitte des Bildes die Xwerte angeschaut werden
            x = x_1d.reshape((-1, 1))
            if displayCalculations:
                print("X Werte neu Shape: ",x.shape)
            #y_pred = model.predict(x)



            #linear regression with numpy
            x_np = x_arr.reshape((1, -11))[0]
            (m, b) = np.polyfit(x_np, y_arr, 1)
            #print(m,b)
            
            if (m < 0.5 and m > -0.5):
                return 0, image
            yp = np.polyval([m, b], x_1d)
            #print(yp)


            #y_pred = model.coef_[0]*x + model.coef_[1]*xfit**2 + model.intercept_
            #y_pred = model.coef_[0]*x + model.intercept_
            for idx,x_element in enumerate(x):
                if show:
                    if idx < 10 and displayCalculations:
                        print(x[idx],yp[idx])
                    #sklearn regression
                    #cv2.circle(image, (int(x[idx]),int(y_pred[idx])),10,(255,0,0))
                    #numpy regression
                    cv2.circle(image, (int(x[idx]),int(yp[idx])),10,(0,255,0))
            if show:
                x1 = x[0][0]
                y1 = int(yp[0])
                x2 = x[-1][0]
                y2 = int(yp[-1])
                cv2.line(image,(x1,y1),(x2,y2),(255,255,0),2)


            #draw middle line at center
            
            middle_screen_y = image.shape[0]//2
            middle_screen_x = (middle_screen_y - b)/m
            #middle_screen_x = (middle_screen_y - model.intercept_)/model.coef_[0]
            #middle_point = (int(middle_screen_x),int(middle_screen_y))
            diff_line = image.shape[1]//2 - middle_screen_x

            if displayCalculations:
                print("image shape: ", image.shape)
                print("middle_screen_y: ",middle_screen_y)
                print("middle_screen_x: ",middle_screen_x)
                #cv2.circle(image,middle_point,5,(255,255,255),5)  
                
                #calc distance between line and center at center height
                print("diff line: ",diff_line)
            if show:
                cv2.circle(image,(image.shape[1]//2,image.shape[0]//2),5,(255,255,255),5)  
        if not show:
            image = None
            
    if diff_line == None:
        return 0,image
    else:
        return diff_line,image