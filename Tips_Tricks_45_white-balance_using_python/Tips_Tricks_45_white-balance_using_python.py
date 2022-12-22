# https://youtu.be/Z0-iM37wseI
"""
White balancing images using two different approaches:
    1. Gray-world algorithm
    2. White patch reference
    
Last tested on:
    python 3.7.11
    opencv 4.5.4
    numpy 1.20.3
"""

import cv2 
import numpy as np

import sys
print("Python version is ", sys.version)

print("opencv version is: ", cv2.__version__)
print("numpy version is: ", np.__version__)

# 1. Gray-world algorithm based white balance

# It assumes that average pixel value is neutral gray (128) because of 
# good distribution of colors. So we can estimate pixel color by looking 
# at the average color. 

# Try the holidays.png image first. Then try lake.png
#This approach works fine on the holidays image but the results from lake
#image can be better. We will use white patch reference for that image later. 

img = cv2.imread('lake.png') #holidays.png , lake.png
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Gray-world algorithm based white balance
#We will convert the image to LAB color space: L for lightness, A for Red/Green and B for Blue/Yellow
#We will calculate the mean color values in A and B channels. 
# Then, subtract 128 (mid gray) from the means and normalize the L channel by
# multiplying with this difference. 
#Finally, subtract this value from A and B channels. 
#Youc can add a multiplication factor to increase/decrease the overall brightness
#of each of the A or B channels. (Here, I added 1.2 as the multiplier)
def GW_white_balance(img):
    img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(img_LAB[:, :, 1])
    avg_b = np.average(img_LAB[:, :, 2])
    img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
    balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2BGR)
    return balanced_image

image_gw_balanced = GW_white_balance(img)

cv2.imshow("Image", img)
cv2.imshow("Image GW Balanced", image_gw_balanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.imwrite("color-balanced-holidays.png", image_gw_balanced)



#### Very good result result from holidays.png image but not great from lake.png
# possibly because there is no good distribution of colors in this image

# 2. White patch reference 

#How about picking a patch of image that is supposed to be white
# and using it as reference to rescale each channel in the image
#You can define a patch if you know the coordinates. How to identify the coordinates?

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x,y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', img)
 
    # checking for right mouse clicks    
    if event==cv2.EVENT_RBUTTONDOWN:
 
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
 
        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv2.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x,y), font, 1,
                    (255, 255, 0), 2)
        cv2.imshow('image', img)


# reading the image
img = cv2.imread('lake.png', 1)
clone = img.copy() #Just so we don't end up with a bunch of annotations on our original image

# display a clickable image.
#From the lake image, let us pick a small rectangle from the cloud region
#where it is supposed to be white. 
cv2.imshow('image', img)
# setting mouse handler for the image
# and calling the click_event() function
cv2.setMouseCallback('image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()



######################

#Now we know the coordinates so let us define a patch around region that is supposed to be white
#Please note that we will get the w pixl value first and h next from the above exercise
#If you consider the image shape to be h x w x 3

#Defining a small rectangle. 
h_start, w_start, h_width, w_width = 174, 502, 10, 10 #For lake.png 174, 502, 10, 10 

image = clone
image_patch = image[h_start:h_start+h_width, 
                    w_start:w_start+w_width]

#Get maximum pixel values from each channel (BGR), normalize the original image
#with these max pixel vlaues - assuming the max pixel is white. 
image_normalized = image / image_patch.max(axis=(0, 1))
print(image_normalized.max())
#Some values will be above 1, so we need to clip the values to between 0 and 1
image_balanced = image_normalized.clip(0,1)

cv2.rectangle(clone, (w_start, h_start), (w_start+w_width, h_start+h_width), (0,0,255), 2)
cv2.imshow("Image", image)
cv2.imshow("Image GW Balanced", image_balanced)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Convert to 8 bit before saving
image_balanced_8bit = (image_balanced*255).astype(int)

cv2.imwrite("color-balanced-lake.png", image_balanced_8bit)

################################################

