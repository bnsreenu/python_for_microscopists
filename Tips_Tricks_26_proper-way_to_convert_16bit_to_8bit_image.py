# https://youtu.be/3vj3MTkBhWg
"""
How to properly convert 16 bit image to 8 bit in python

Scientific images often come with higher bit depths than 8 bit. 

To read czi file...
pip install czifile

"""
import sys
print("python version is: ", sys.version)


import czifile
print("czifile version is: ", czifile.__version__)
from matplotlib import pyplot as plt
import numpy as np

img = czifile.imread('Osteosarcoma_01.czi')

print(img.shape)  #7 dimensions
#Time series, scenes, channels, x, y, z, RGB
#IN this example (Osteosarcoma) we have 1 time series, 1 scene, 3 channels and each channel grey image
#size 1104 x 1376

#Let us extract only relevant pixels, all channels in x and y
img_3channel=img[0, 0, :, :, :, 0]
print(img_3channel.shape)

DAPI = (img_3channel[2,:,:]) #Third channel, convert to uint8 from uint16

plt.imshow(DAPI, cmap='gray')


#Direct conversion
import cv2
DAPI_8bit_a=cv2.convertScaleAbs(DAPI)
plt.imshow(DAPI_8bit_a, cmap='gray')  #Saturated pixels. 

from skimage import img_as_ubyte
DAPI_8bit_b=img_as_ubyte(DAPI)
plt.imshow(DAPI_8bit_b, cmap='gray')
print("maximum pixel value is: ", DAPI_8bit_b.max())

############# PROPER WAY(S) ###################
#Normalize then scale to 255 and convert to uint8 - manual
DAPI_8bit_c = (DAPI / DAPI.max()) * 255.
DAPI_8bit_c = np.uint8(DAPI_8bit_c)
plt.imshow(DAPI_8bit_c, cmap='gray')

#Normalize then scale to 255 and convert to uint8 - using opencv
DAPI_8bit_d = cv2.normalize(DAPI, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
plt.imshow(DAPI_8bit_d, cmap='gray')

#Normalize then scale to 255 and convert to uint8 - using skimage
from skimage import exposure, img_as_ubyte
#exposure.rescale_intensity stretches (scales) intensity to min and max 
# of the dtype of the image. In our case: 0 to 65535
DAPI_8bit_e = img_as_ubyte(exposure.rescale_intensity(DAPI))
plt.imshow(DAPI_8bit_e, cmap='gray')

