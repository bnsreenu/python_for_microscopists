#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/TyV-9K_8w20

"""
Learning from the astronomy guys...
A couple of ways to perform image registration
https://image-registration.readthedocs.io/en/latest/image_registration.html
"""
from skimage import io
from image_registration import chi2_shift

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 

#Method 1: chi squared shift
#Find the offsets between image 1 and image 2 using the DFT upsampling method
noise=0.1
xoff, yoff, exoff, eyoff = chi2_shift(image, offset_image, noise, 
                                      return_error=True, upsample_factor='auto')

print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)

from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()

###########################################################################
#Method 2: Cross correlation based shift
#Use cross-correlation and a 2nd order taylor expansion to measure the shift


from skimage import io
from image_registration import cross_correlation_shifts

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 


xoff, yoff = cross_correlation_shifts(image, offset_image)


print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()


############################################################
#Method 3: Optical flow based shift
#takes two images and returns a vector field. 
#For every pixel in image 1 you get a vector showing where it moved to in image 2.

from skimage import io
from image_registration import cross_correlation_shifts

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 


from skimage import registration
flow = registration.optical_flow_tvl1(image, offset_image)

# display dense optical flow
flow_x = flow[1, :, :]
flow_y = flow[0, :, :]


#Let us find the mean of all pixels in x and y and shift image by that amount
#ideally, you need to move each pixel by the amount from flow
import numpy as np
xoff = np.mean(flow_x)
yoff = np.mean(flow_y)


print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()


##################################################

#Method 4: register_translation from skimage.feature (already covered in previous video)
from skimage import io
from image_registration import cross_correlation_shifts

image = io.imread("images/BSE.jpg")
offset_image = io.imread("images/BSE_transl.jpg")
# offset image translated by (-17.45, 18.75) in y and x 


from skimage.feature import register_translation
shifted, error, diffphase = register_translation(image, offset_image, 100)
xoff = -shifted[1]
yoff = -shifted[0]


print("Offset image was translated by: 18.75, -17.45")
print("Pixels shifted by: ", xoff, yoff)


from scipy.ndimage import shift
corrected_image = shift(offset_image, shift=(xoff,yoff), mode='constant')

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(image, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(offset_image, cmap='gray')
ax2.title.set_text('Offset image')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(corrected_image, cmap='gray')
ax3.title.set_text('Corrected')
plt.show()