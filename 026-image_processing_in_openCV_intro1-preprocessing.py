#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=-Qnb8Wv2p1c


# Image smoothing, denoising
# Averaging, gaussian blurring, median, bilateral filtering
#OpenCV has a function cv2.filter2D(), which convolves whatever kernel we define with the image.

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('images/BSE_Google_noisy.jpg', 1)
kernel = np.ones((5,5),np.float32)/25
filt_2D = cv2.filter2D(img,-1,kernel)    #Convolution using the kernel we provide
blur = cv2.blur(img,(5,5))   #Convolution with a normalized filter. Same as above for this example.
blur_gaussian = cv2.GaussianBlur(img,(5,5),0)  #Gaussian kernel is used. 
median_blur = median = cv2.medianBlur(img,5)  #Using kernel size 5. Better on edges compared to gaussian.
bilateral_blur = cv2.bilateralFilter(img,9,75,75)  #Good for noise removal but retain edge sharpness. 


cv2.imshow("Original", img)
cv2.imshow("2D filtered", filt_2D)
cv2.imshow("Blur", blur)
cv2.imshow("Gaussian Blur", blur_gaussian)
cv2.imshow("Median Blur", median_blur)
cv2.imshow("Bilateral", bilateral_blur)
cv2.waitKey(0)          
cv2.destroyAllWindows() 

#############################################################

#Edge detection:
    
import cv2
import numpy as np

img = cv2.imread("images/Neuron.jpg", 0)
edges = cv2.Canny(img,100,200)   #Image, min and max values

cv2.imshow("Original Image", img)
cv2.imshow("Canny", edges)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

#########################################################

