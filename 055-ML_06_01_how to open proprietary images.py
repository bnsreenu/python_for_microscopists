#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=rCcQrAMkmQ4


"""
@author: Sreenivas Bhattiprolu

#First google search for czi file and python
You will see a few libraries. 
Let us click on the first one, czifile
This is from Christoph Gohlke. 
It shows how to install... pip install
I put ! before pip from the terminal because it indicates running as a shell command.
Almost like opening windows terminal and typing the pip command. 
"""

import czifile



img = czifile.imread('Osteosarcoma_01.czi')

print(img.shape)  #7 dimensions
#Time series, scenes, channels, x, y, z, RGB
#IN this example (Osteosarcoma) we have 1 time series, 1 scene, 3 channels and each channel grey image
#size 1104 x 1376

#Let us extract only relevant pixels, all channels in x and y
img1=img[0, 0, :, :, :, 0]
print(img1.shape)

#Next, let us extract each channel image.
img2=img1[0,:,:]  #First channel
img3=img1[1,:,:] #Second channel
img4=img1[2,:,:] #Third channel
#io.imshow(img4)

import matplotlib.pyplot as plt
plt.imshow(img2, cmap='hot')

#Olympus images, similar way https://pypi.org/project/oiffile/
