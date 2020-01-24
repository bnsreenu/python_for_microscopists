#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=WQK_oOWW5Zo


#Morphological transformations
#Let us take the image from above that we got from OTSU and try various filters.

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("images/BSE_Google_noisy.jpg", 0)

ret,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)   # 3x3 kernel with all ones. 
erosion = cv2.erode(th,kernel,iterations = 1)  #Erodes pixels based on the kernel defined

dilation = cv2.dilate(erosion,kernel,iterations = 1)  #Apply dilation after erosion to see the effect. 

#Erosion followed by dilation can be a single operation called opening
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)  # Compare this image with the previous one

#Closing is opposit, dilation followed by erosion.
closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

#Morphological gradient. This is the difference between dilation and erosion of an image
gradient = cv2.morphologyEx(th, cv2.MORPH_GRADIENT, kernel)

#It is the difference between input image and Opening of the image. 
tophat = cv2.morphologyEx(th, cv2.MORPH_TOPHAT, kernel)

#It is the difference between the closing of the input image and input image.
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Original Image", blackhat)
cv2.waitKey(0)          
cv2.destroyAllWindows() 




########################################
