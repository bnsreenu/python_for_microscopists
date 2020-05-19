#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"


# https://youtu.be/jZ97DtM3YMQ


"""
@author: Sreenivas Bhattiprolu

https://scikit-image.org/docs/dev/api/skimage.measure.html

https://pypi.org/project/sewar/
https://sewar.readthedocs.io/en/latest/_modules/sewar/full_ref.html#ergas

"""

import cv2
import numpy as np
from sewar import full_ref
from skimage import measure


#Reference and image to be compared must be of the same size
ref_img = cv2.imread("images/BSE.jpg", 1)
img = cv2.imread("images/BSE_25sigma_noisy.jpg", 1)

################################################################
#skimage tools
#Mean square error

mse_skimg = measure.compare_mse(ref_img, img)
print("MSE: based on scikit-image = ", mse_skimg)

#Same as PSNR available in sewar
psnr_skimg = measure.compare_psnr(ref_img, img, data_range=None)
print("PSNR: based on scikit-image = ", psnr_skimg)

#Normalized root mean squared error
rmse_skimg = measure.compare_nrmse(ref_img, img)
print("RMSE: based on scikit-image = ", rmse_skimg)


###############################################################
#ERGAS Global relative error
"""calculates global relative error 
GT: first (original) input image.
P: second (deformed) input image.
r: ratio of high resolution to low resolution (default=4).
ws: sliding window size (default = 8).

	:returns:  float -- ergas value.
	"""
ergas_img = full_ref.ergas(ref_img, img, r=4, ws=8)
print("EGRAS: global relative error = ", ergas_img)



####################################################################
#Multiscale structural similarity index
"""calculates multi-scale structural similarity index (ms-ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param weights: weights for each scale (default = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).
	:param ws: sliding window size (default = 11).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- ms-ssim value.
	"""
msssim_img=full_ref.msssim(ref_img, img, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None)

print("MSSSIM: multi-scale structural similarity index = ", msssim_img)


##############################################################################
#PSNR
"""calculates peak signal-to-noise ratio (psnr).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param MAX: maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  float -- psnr value in dB.
	"""
psnr_img=full_ref.psnr(ref_img, img, MAX=None)

print("PSNR: peak signal-to-noise ratio = ", psnr_img)


##########################################################################
#PSNRB: Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)
"""Calculates PSNR with Blocking Effect Factor for a given pair of images (PSNR-B)

	:param GT: first (original) input image in YCbCr format or Grayscale.
	:param P: second (corrected) input image in YCbCr format or Grayscale..
	:return: float -- psnr_b.
	"""
#psnrb_img = full_ref.psnrb(ref_img, img)

#print("PSNRB: peak signal-to-noise ratio with blocking effect = ", psnrb_img)

#######################################################################
#relative average spectral error (rase)
"""calculates relative average spectral error (rase).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- rase value.
	"""
RASE_img = full_ref.rase(ref_img, img, ws=8)
#print("RASE: relative average spectral error = ", RASE_img)


######################################################################
#RMSE
"""calculates root mean squared error (rmse).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- rmse value.
	"""
rmse_img = full_ref.rmse(ref_img, img)
print("RMSE: root mean squared error = ", rmse_img)



######################################################################
#root mean squared error (rmse) using sliding window
"""calculates root mean squared error (rmse) using sliding window.

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  tuple -- rmse value,rmse map.	
	"""
rmse_sw_img = full_ref.rmse_sw(ref_img, img, ws=8)
#print("RMSE_SW: root mean squared error with sliding window = ", rmse_sw_img)


#########################################################################
#calculates spectral angle mapper (sam).
"""calculates spectral angle mapper (sam).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.

	:returns:  float -- sam value.
	"""
ref_sam_img = full_ref.sam(ref_img, img)
print("REF_SAM: spectral angle mapper = ", ref_sam_img)


######################################################################
#Spatial correlation coefficient
full_ref.scc(ref_img, img, win=[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], ws=8)

#Structural similarity index
"""calculates structural similarity index (ssim).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).
	:param K1: First constant for SSIM (default = 0.01).
	:param K2: Second constant for SSIM (default = 0.03).
	:param MAX: Maximum value of datarange (if None, MAX is calculated using image dtype).

	:returns:  tuple -- ssim value, cs value.
	"""
ssim_img = full_ref.ssim(ref_img, img, ws=11, K1=0.01, K2=0.03, MAX=None, fltr_specs=None, mode='valid')
print("SSIM: structural similarity index = ", ssim_img)

##############################################################################
#Universal image quality index
"""calculates universal image quality index (uqi).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- uqi value.
	"""
UQI_img = full_ref.uqi(ref_img, img, ws=8)
print("UQI: universal image quality index = ", UQI_img)

##############################################################################
#Pixel Based Visual Information Fidelity (vif-p)
"""calculates Pixel Based Visual Information Fidelity (vif-p).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param sigma_nsq: variance of the visual noise (default = 2)

	:returns:  float -- vif-p value.
	"""
VIFP_img = full_ref.vifp(ref_img, img, sigma_nsq=2)
print("VIFP: Pixel Based Visual Information Fidelity = ", VIFP_img)






