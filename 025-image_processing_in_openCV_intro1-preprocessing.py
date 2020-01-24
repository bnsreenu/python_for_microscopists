#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=yj40XoUqo70


######### Using openCV #########

#to install open CV : pip install opencv-python
#to import the package you need to use import cv2

#openCV is a library of programming functions mainly aimed at computer vision.
#Very good for images and videos, especially real time videos.
#It is used extensively for facial recognition, object recognition, motion tracking,
#optical character recognition, segmentation, and even for artificial neural netwroks. 

#Useful preprocessing steps for image processing, for example segmentation. 
#1. SPlit & Merge channels
#2. Scaling / resizing
#3. Denoising / smoothing
#4. Edge detection
#5. Enhancing images. using histogram equalization

###################################
#Pixel values, split and merge channels, 

import cv2

grey_img = cv2.imread("images/RGBY.jpg", 0) 
img = cv2.imread("images/RGBY.jpg", 1)   #Color is BGR not RGB

print(img.shape)     #(586, 415, 3)
print("Top left", img[0,0])    #Top left pixel
print("Top right", img[0, 400])  # Top right
print("Bottom Left", img[580, 0]) # Bottom left
print("Bottom right", img[580, 400])  # Bottom right

#cv2.imshow("color pic", img)
#cv2.waitKey(0)          
#cv2.destroyAllWindows() 

#Split and merging channels
#Show individual color channels in the image
blue = img[:, :, 0]   #Show only blue pic. (BGR so B=0)
green = img[:, :, 1]  #Show only green pixels
red = img[:, :, 2]  #red only

"""
cv2.imshow("red pic", red)
cv2.waitKey(0)          
cv2.destroyAllWindows() 
"""

#Or split all channels at once

b,g,r = cv2.split(img)

cv2.imshow("green pic", g)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

#to merge each image into bgr

img_merged = cv2.merge((b,g,r))

cv2.imshow("merged pic", img_merged)
cv2.waitKey(0)          
cv2.destroyAllWindows() 


######################
#Basic image operations
# Scaling, 
#https://docs.opencv.org/3.3.1/da/d6e/tutorial_py_geometric_transformations.html

import cv2

img = cv2.imread("images/monkey.jpg", 1)   #Color is BGR not RGB

#use cv2.resize. Can specify size or scaling factor.
#Inter_cubic or Inter_linear for zooming.
#Use INTER_AREA for shrinking
#Following xample zooms by 2 times.

resized = cv2.resize(img, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)

cv2.imshow("original pic", img)
cv2.imshow("resized pic", resized)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

