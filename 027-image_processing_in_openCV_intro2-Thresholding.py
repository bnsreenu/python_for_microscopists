#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=XfDkg3z3BCg



#Image enhancements: 
# Sometimes microscope images lack contrast, they appear to be washed out but they still contain information.
# (Show scratch assay and alloy images)
# We can mathematically process these images and make them look good,
#more importantly, get them ready for segmentation
#
#Histogram equalization is a good way to stretch the histogram and thus improve the image.  


import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("images/Alloy.jpg", 0)
equ = cv2.equalizeHist(img)

plt.hist(equ.flat, bins=100, range=(0,100))

cv2.imshow("Original Image", img)
cv2.imshow("Equalized", equ)


#Histogram Equalization considers the global contrast of the image, may not give good results.
#Adaptive histogram equalization divides images into small tiles and performs hist. eq.
#Contrast limiting is also applied to minimize aplification of noise.
#Together the algorithm is called: Contrast Limited Adaptive Histogram Equalization (CLAHE)

# Start by creating a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
cl1 = clahe.apply(img)

cv2.imshow("CLAHE", cl1)

cv2.waitKey(0)          
cv2.destroyAllWindows() 


######################################################
###################################################
#Image thresholding

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/Alloy.jpg", 0)

#Adaptive histogram equalization using CLAHE to stretch the histogram. 
#Contrast Limited Adaptive Histogram Equalization covered in the previous tutorial. 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)
plt.hist(clahe_img.flat, bins =100, range=(0,255))

#Thresholding. Creates a uint8 image but with binary values.
#Can use this image to further segment.
#First argument is the source image, which should be a grayscale image.
#Second argument is the threshold value which is used to classify the pixel values. 
#Third argument is the maxVal which represents the value to be given to the thresholded pixel.

ret,thresh1 = cv2.threshold(clahe_img,185,150,cv2.THRESH_BINARY)  #All thresholded pixels in grey = 150
ret,thresh2 = cv2.threshold(clahe_img,185,255,cv2.THRESH_BINARY_INV) # All thresholded pixels in white

cv2.imshow("Original", img)
cv2.imshow("Binary thresholded", thresh1)
cv2.imshow("Inverted Binary thresholded", thresh2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

############################################
#OTSU Thresholding, binarization
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/Alloy.jpg", 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  #Define tile size and clip limit. 
clahe_img = clahe.apply(img)

plt.hist(clahe_img.flat, bins =100, range=(0,255))

# binary thresholding
ret1,th1 = cv2.threshold(clahe_img,185,200,cv2.THRESH_BINARY)

# Otsu's thresholding, automatically finds the threshold point. 
#Compare wth above value provided by us (185)
ret2,th2 = cv2.threshold(clahe_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


cv2.imshow("Otsu", th2)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

# If working with noisy images
# Clean up noise for better thresholding
# Otsu's thresholding after Gaussian filtering. Canuse median or NLM for beteer edge preserving

import cv2
import matplotlib.pyplot as plt

img = cv2.imread("images/Alloy_noisy.jpg", 0)

blur = cv2.GaussianBlur(clahe_img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


plt.hist(blur.flat, bins =100, range=(0,255))
cv2.imshow("OTSU Gaussian cleaned", th3)
cv2.waitKey(0)          
cv2.destroyAllWindows() 


