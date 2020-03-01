#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/1GUgD2SBl9A

"""
Spyder Editor

scipy.signal.convolve2d - https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
scipy.ndimage.filters.convolve
cv2.filter2D - https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d

"""
import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage.filters import convolve
from skimage import io, img_as_float


img_gaussian_noise = img_as_float(io.imread('images/BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('images/BSE_salt_pepper.jpg', as_gray=True))

img = img_salt_pepper_noise

#convolving the image with a normalized box filter. 
#It simply takes the average of all the pixels under kernel area 
#and replaces the central element with this average. 
#Can also use cv2.blur() to directly apply the convolution rather than defining 
#kernel and applying it as a filter (or convolution) separately. 
kernel = np.ones((5,5),np.float32)/25    #Averaging filter with 5x5 kernel
#Normalize by dividing with 25 so all numbers add to 1
gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                [1/8, 1/4, 1/8],
                [1/16, 1/8, 1/16]])

laplacian = np.array([[0.,1,0], [1,-4,1], [0,1,0]])
gabor = cv2.getGaborKernel((5, 5), 1.4, 45, 5, 1)

"""
Results from all approaches would be the same..
Remember that if the padding (border) is not done then the output image
and values would be different

"""

conv_using_cv2 = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT) 
# when ddepth=-1, the output image will have the same depth as the source
#example, if input is float64 then output will also be float64
# BORDER_CONSTANT - Pad the image with a constant value (i.e. black or 0)
#BORDER_REPLICATE: The row or column at the very edge of the original is replicated to the extra border.

conv_using_scipy = convolve2d(img, kernel, mode='same')
#mode ="same" - pads image so the output is same as input

conv_using_scipy2 = convolve(img, kernel, mode='constant', cval=0.0)
#mode=constant adds a constant value at the borders. 


cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using scipy", conv_using_scipy)
cv2.imshow("Using scipy2", conv_using_scipy2)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

