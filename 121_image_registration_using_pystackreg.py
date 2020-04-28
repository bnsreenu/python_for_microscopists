#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

#https://youtu.be/5FEr5SiXB1g
"""
Image registration using pystackreg
https://pypi.org/project/pystackreg/

pip install pystackreg

pyStackReg is used to align (register) one or more images to a 
common reference image, as is required usually in time-resolved fluorescence 
or wide-field microscopy. 

It uses functionality from imageJ

pyStackReg provides the following four types of distortion:

1. translation
2. rigid body (translation + rotation)
3. scaled rotation (translation + rotation + scaling)
4. affine (translation + rotation + scaling + shearing)
5. bilinear (non-linear transformation; does not preserve straight lines)

"""

from pystackreg import StackReg
from skimage import io
from matplotlib import pyplot as plt


#The following example opens two different files and registers them 
#using all different possible transformations

#load reference and "moved" image
ref_img = io.imread('images/for_alignment/shale_for_alignment00.tif')
offset_img = io.imread('images/for_alignment/shale_for_alignment01.tif')

#Translational transformation
sr = StackReg(StackReg.TRANSLATION)
out_tra = sr.register_transform(ref_img, offset_img)
plt.imshow(out_tra, cmap='gray')


#Rigid Body transformation
sr = StackReg(StackReg.RIGID_BODY)
out_rot = sr.register_transform(ref_img, offset_img)

#Scaled Rotation transformation
#sr = StackReg(StackReg.SCALED_ROTATION)
#out_sca = sr.register_transform(ref_img, offset_img)

#Affine transformation
sr = StackReg(StackReg.AFFINE)
out_aff = sr.register_transform(ref_img, offset_img)

#Bilinear transformation
#sr = StackReg(StackReg.BILINEAR)
#out_bil = sr.register_transform(ref_img, offset_img)


#Plotting a few outputs

from matplotlib import pyplot as plt
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(ref_img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(out_tra, cmap='gray')
ax2.title.set_text('Translation')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(out_rot, cmap='gray')
ax3.title.set_text('Rigid Body')
ax3 = fig.add_subplot(2,2,4)
ax3.imshow(out_aff, cmap='gray')
ax3.title.set_text('Affine')
plt.show()



#Looking at single image doesn't make sense for this example.
#Let us look at a stack


#############################################
#To create a tiff stack image from individual images

import glob
import tifffile

with tifffile.TiffWriter('images/my_image_stack.tif') as stack:
    for filename in glob.glob('images/for_alignment/*.tif'):
        stack.save(tifffile.imread(filename))

####################################################################

#How to register and transform a whole stack:

from pystackreg import StackReg
from skimage import io

img0 = io.imread('images/my_image_stack.tif') # 3 dimensions : frames x width x height

sr = StackReg(StackReg.RIGID_BODY)

# register each frame to the previous (already registered) one
# this is what the original StackReg ImageJ plugin uses
out_previous = sr.register_transform_stack(img0, reference='previous')

#To save the output to a tiff stack image
#First convert float values to int
import numpy
out_previous_int = out_previous.astype(numpy.int8)

#Using tifffile to save the stack into a single tif
import tifffile
tifffile.imsave('images/my_aligned_stack.tif', out_previous_int)


# register to first image
out_first = sr.register_transform_stack(img0, reference='first')

# register to mean image
out_mean = sr.register_transform_stack(img0, reference='mean')

# register to mean of first 10 images
out_first10 = sr.register_transform_stack(img0, reference='first', n_frames=10)

# calculate a moving average of 10 images, then register the moving average to the mean of
# the first 10 images and transform the original image (not the moving average)
out_moving10 = sr.register_transform_stack(img0, reference='first', n_frames=10, moving_average = 10)
