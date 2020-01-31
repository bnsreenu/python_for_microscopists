#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=yUATC9tt7OM


"""
@author: Sreenivas Bhattiprolu

What are features? 


"""

#A few generic filters...


import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pandas as pd

img = cv2.imread('scratch.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img2 = img.reshape(-1)
df = pd.DataFrame()
df['Original Image'] = img2


entropy_img = entropy(img, disk(1))

entropy1 = entropy_img.reshape(-1)
df['Entropy'] = entropy1
#print(df)

"""
cv2.imshow('entropy', entropy_img)
cv2.waitKey()
cv2.destroyAllWindows()
"""

import cv2
from scipy import ndimage as nd
#import numpy as np
"""
img = cv2.imread('Yeast_Cells.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
"""
gaussian_img = nd.gaussian_filter(img, sigma=3)

gaussian_img1 = gaussian_img.reshape(-1)
df['Gaussian s3'] = gaussian_img1


from skimage.filters import sobel
sobel_img = sobel(img)

sobel1 = sobel_img.reshape(-1)
df['Sobel'] = sobel1
print(df.head())

cv2.imshow('Original Image', img)
cv2.imshow('gaussian', sobel_img)
cv2.waitKey(5000)
cv2.destroyAllWindows()

