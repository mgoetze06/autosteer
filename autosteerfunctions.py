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
    def computePolyLoss(y,y_pred,degree,param):
        loss = np.sum(np.sqrt((y - y_pred)**2) + param*degree**2)
        return loss
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
                    #cv2.circle(image, (x,y),10,(0,0,0))
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
            use_xarr_from_detection = True
            if use_xarr_from_detection:
                x = x_arr
            else:
                x_1d = np.arange(600,1100) #hier muss theoertisch nur in der Mitte des Bildes die Xwerte angeschaut werden
                x = x_1d.reshape((-1, 1))
            if displayCalculations:
                print("X Werte neu Shape: ",x.shape)
            #y_pred = model.predict(x)



            #linear regression with numpy
            x_np = x_arr.reshape((1, -1))[0]
            degree = 1
            coeffs = np.polyfit(x_np, y_arr, degree)
            polymodel = np.poly1d(coeffs)
            one_dim_regression = True
            if one_dim_regression:
                #numpy linear reg; degree 1
                m, b = coeffs
                if (m < 0.5 and m > -0.5): #ignore these slopes
                    return 0, image, 0, 0
                yp = np.polyval([m, b], x_np)
                #print(yp)
            else:
                yp = polymodel(x_np)
            
            #print("y fitting: ",y_arr.reshape(-1,1).shape)
            #print("y neu: ",yp.shape)
            #print("degree: ",degree)
            regression_error = computePolyLoss(y_arr.reshape(-1,1),yp,degree,1)
            #print("regression error (loss): ",regression_error)

            #y_pred = model.coef_[0]*x + model.coef_[1]*xfit**2 + model.intercept_
            #y_pred = model.coef_[0]*x + model.intercept_
            for idx,x_element in enumerate(x_np):
                if show:
                    if idx < 10 and displayCalculations:
                        print(x_np[idx],yp[idx])
                    #sklearn regression
                    #cv2.circle(image, (int(x[idx]),int(y_pred[idx])),10,(255,0,0))
                    #numpy regression
                    cv2.circle(image, (int(x_np[idx]),int(yp[idx])),10,(0,0,0))
            if show:
                x1 = x[0][0]
                y1 = int(yp[0])
                x2 = x[-1][0]
                y2 = int(yp[-1])
                cv2.line(image,(x1,y1),(x2,y2),(255,255,0),2)


            #draw middle line at center
            if one_dim_regression:
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
        return 0,image,0,0
    else:
        return diff_line,image,regression_error,m

def rectangle_roi_center(img,width_factor,height_factor,y_offset,x_offset):
    #print(img.shape)
    rect_height = round(width_factor * img.shape[0])
    rect_width = round(height_factor * img.shape[1])
    x_mid = img.shape[1]//2
    y_mid = img.shape[0]//2
    #print(y_mid,x_mid)
    #print(rect_height,rect_width)
    #y_offset = 150
    p1 = y_mid-rect_height+y_offset
    p2 = y_mid+rect_height+y_offset
    p3 = x_mid-rect_width+x_offset
    
    p4 = x_mid+rect_width+x_offset
    #print(p1,p2,p3,p4)
    
    roi = img[p1:p2,p3:p4]
    
    #distance_from_center = 0
    
    return roi

def get_schwad_distance_from_center(img,line_index,show):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_gray, cmap="gray")
    img_gray_res = exposure.rescale_intensity(img_gray,in_range=(10,100),out_range='uint8')

    kernel_size = 35
    blur_gray = cv2.GaussianBlur(img_gray_res,(kernel_size, kernel_size),0)

    line = blur_gray[line_index]
    #print(line)
    #plt.plot(line)

    avg = np.convolve(line, np.ones(kernel_size)/kernel_size, mode='valid')
    #plt.plot(avg)
    #plt.plot(np.diff(line))
    #print(np.argmax(np.gradient(line)))
    #print(np.argmin(np.gradient(line)))

    up = np.argmax(np.gradient(avg))
    down = np.argmin(np.gradient(avg))

    bild_mitte = len(line)//2

    schwad_breite = abs(up - down)
    schwad_mitte = min(up,down) + schwad_breite // 2


    fehler = schwad_mitte - bild_mitte
    #cv2.line(img, (10, line_index), (100, line_index), (0, 255, 0), thickness=2)
    cv2.line(img, (bild_mitte, 10), (bild_mitte, 100), (255, 255, 0), thickness=2)
    cv2.line(img, (schwad_mitte, 10), (schwad_mitte, 100), (0, 255, 0), thickness=2)
    cv2.line(img, (up, 10), (up, 100), (0, 255, 0), thickness=1)
    cv2.line(img, (down, 10), (down, 100), (0, 255, 0), thickness=1)
    cv2.line(img, (up, line_index), (down, line_index), (0, 255, 0), thickness=1)
    
    if show:
        #plt.plot(line)
        plt.plot(avg)
        plt.vlines(schwad_mitte,75,150,colors='r')
        plt.vlines(up,75,150,colors='b')
        plt.vlines(down,75,150,colors='b')
        plt.vlines(len(line)//2,75,150)
    #print(schwad_breite,schwad_mitte)
    #print("Fehler: ",fehler)
    return fehler,img