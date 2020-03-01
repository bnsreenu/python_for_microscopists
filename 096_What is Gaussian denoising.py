#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/xCHbcVUCYBI

"""

cv2.filter2D - https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=filter2d#filter2d
cv2.GaussianBlur - https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/
skimage.filters.gaussian - https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.gaussian
"""
import cv2
import numpy as np
from skimage import io, img_as_float
from skimage.filters import gaussian

img_gaussian_noise = img_as_float(io.imread('images/BSE_25sigma_noisy.jpg', as_gray=True))
img_salt_pepper_noise = img_as_float(io.imread('images/BSE_salt_pepper.jpg', as_gray=True))

img = img_gaussian_noise

gaussian_kernel = np.array([[1/16, 1/8, 1/16],   #3x3 kernel
                [1/8, 1/4, 1/8],
                [1/16, 1/8, 1/16]])



conv_using_cv2 = cv2.filter2D(img, -1, gaussian_kernel, borderType=cv2.BORDER_CONSTANT) 
# when ddepth=-1, the output image will have the same depth as the source
#example, if input is float64 then output will also be float64
# BORDER_CONSTANT - Pad the image with a constant value (i.e. black or 0)
#BORDER_REPLICATE: The row or column at the very edge of the original is replicated to the extra border.

gaussian_using_cv2 = cv2.GaussianBlur(img, (3,3), 0, borderType=cv2.BORDER_CONSTANT)

gaussian_using_skimage = gaussian(img, sigma=1, mode='constant', cval=0.0)
#sigma defines the std dev of the gaussian kernel. SLightly different than 
#how we define in cv2


cv2.imshow("Original", img)
cv2.imshow("cv2 filter", conv_using_cv2)
cv2.imshow("Using cv2 gaussian", gaussian_using_cv2)
cv2.imshow("Using skimage", gaussian_using_skimage)
#cv2.imshow("Using scipy2", conv_using_scipy2)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

