#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/aJrG8IH81SY

"""

http://www.cs.tut.fi/~foi/papers/ICIP2019_Ymir.pdf

"""

import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import cv2

noisy_img = img_as_float(io.imread("images/BSE_25sigma_noisy.jpg", as_gray=True))

BM3D_denoised_image = bm3d.bm3d(noisy_img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)

"""
bm3d library is not well documented yet, but looking into source code....
sigma_psd - noise standard deviation
stage_arg: Determines whether to perform hard-thresholding or Wiener filtering.
stage_arg = BM3DStages.HARD_THRESHOLDING or BM3DStages.ALL_STAGES (slow but powerful)
All stages performs both hard thresholding and Wiener filtering. 
"""

cv2.imshow("Original", noisy_img)
cv2.imshow("Denoised", BM3D_denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()