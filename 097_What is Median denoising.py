#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/StX_1iEO3ck

"""
Spyder Editor

cv2.medianBlur - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
skimage.filters.median - https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.median

See how median is much better at cleaning salt and pepper noise compared to Gaussian
"""
import cv2
import numpy as np
from scipy.ndimage.filters import convolve
from skimage import io
from skimage.filters import median

#img = io.imread('images/einstein.jpg', as_gray=True)

#Needs 8 bit, not float.
img_gaussian_noise = cv2.imread('images/BSE_25sigma_noisy.jpg', 0)
img_salt_pepper_noise = cv2.imread('images/BSE_salt_pepper.jpg', 0)

img = img_salt_pepper_noise


median_using_cv2 = cv2.medianBlur(img, 3)

from skimage.morphology import disk
median_using_skimage = median(img, disk(3), mode='constant', cval=0.0)


cv2.imshow("Original", img)
cv2.imshow("cv2 median", median_using_cv2)
cv2.imshow("Using skimage median", median_using_skimage)

cv2.waitKey(0)          
cv2.destroyAllWindows() 

