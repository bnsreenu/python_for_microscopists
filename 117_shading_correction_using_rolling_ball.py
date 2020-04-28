#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/hy5PlXX-CU0


"""
1st approach: Perform CLAHE
# Equalize light by performing CLAHE on the Luminance channel
# The equalize part alreay covered as aprt of previous tutorials about CLAHE
# This kind of works but you can still see shading after the correction.

2nd approach:
Apply rolling ball background subtraction

"""
import cv2
import numpy as np

img = cv2.imread("images/Alloy_gradient.jpg", 1)

lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_img)


clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
clahe_img = clahe.apply(l)
CLAHE_img = cv2.merge((clahe_img,a,b))

corrected_image = cv2.cvtColor(CLAHE_img, cv2.COLOR_LAB2BGR)

cv2.imshow("Original image", img)
cv2.imshow("Corrected image", corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

############################################################
"""
#2nd method
# https://pypi.org/project/opencv-rolling-ball/
# 
# pip install opencv-rolling-ball
# Only works with 8 bit grey

A local background value is determined for every pixel by averaging over a 
very large ball around the pixel. This value is then subtracted from 
the original image, removing large spatial variations of the 
background intensities. The radius should be set to at least the size of the 
largest object that is not part of the background.
"""

import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt

img = cv2.imread("images/Alloy_gradient.jpg", 0)
radius=30
final_img, background = subtract_background_rolling_ball(img, radius, light_background=True,
                                     use_paraboloid=False, do_presmooth=True)


#optionally perform CLAHE to equalize histogram for better segmentation
#otherwise the image may appear washedout. 

clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
clahe_img = clahe.apply(final_img)


#cv2.imshow("Original image", img)
cv2.imshow("Background image", background)
cv2.imshow("AFter background subtraction", final_img)
cv2.imshow("After CLAHE", clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()