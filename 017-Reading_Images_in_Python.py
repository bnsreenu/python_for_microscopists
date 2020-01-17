#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=52pMFnkDU-4

"""
Many ways to open images in Python.
PIL
matplotlib
skimage
openCV
other libraries to open propriatery images like czi, OME-TIFF
"""


"""
############################################################################
######### Using PIL, Python Imaging Library #########

#Pillow is an image manipulation and processing library
#You can use pillow to crop, resize images and to do basic filtering.
#For advanced tasks that require computer vision or machine elarning we have other packages.
#such as openCV, scikit image and scikit learn. 

# to install pillow, pip install Pillow 
# to import the package you need to use import PIL
"""


from PIL import Image 
import numpy as np   #Use numpy to convert images to arrays

# Read image 
img = Image.open("images/test_image.jpg") #Not a numpy array
print(type(img))

# Output Images 
img.show() 

# prints format of image 
print(img.format) 
  
# prints mode of image 
print(img.mode) 

#PIL is not by default numpy array but can convert PIL image to numpy array. 
img1 = np.asarray(img)
print(type(img1))

"""
###############################################################################
######### Using Matplotlib #########
#Matplotlib is a plotting library for the Python programming language
#Pyplot is a Matplotlib module which provides a MATLAB-like interface
#Pyplot is commonly used not just to generate plots and graphs but also to visualize images.
#because visualizing images is nothing but plotting data in 2D. 

# to install matplotlib, pip install matplotlib 
# to import the package you need to use import matplotlib

"""

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

img = mpimg.imread("images/test_image.jpg")  #this is a numpy array
print(type(img))
print(img)

print(img.shape)

plt.imshow(img)
plt.colorbar()   #Puts a color bar next to the image. 

"""
#############################################################################
######Using scikit image ############
# to install matplotlib, pip install scikit-image 
# to import the package you need to use import skimage
#scikit image is an image processing library that includes alforithms for
#segmentation, geometric transformation, color space manipulation, analysis, filtering, 
#feature detection, and more.
#A very good package for traditional machine learning, using Random forest or SVM
"""

from skimage import io, img_as_float, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt 


image = img_as_float(io.imread("images/test_image.jpg"))

#image2 = io.imread("images/test_image.jpg").astype(np.float)
#avoid using astype as it violates assumptions about dtype range.
#for example float should range from 0 to 1 (or -1 to 1) but if you use 
#astype to convert to float, the values do not lie between 0 and 1. 
#print(image.shape)
#plt.imshow(img)

print(image)

#print(image2)
#image8byte = img_as_ubyte(image)
#print(image8byte)

#End of Skimage

"""
#################################################################################
######### Using openCV #########


#to install open CV : pip install opencv-python
#to import the package you need to use import cv2
#openCV is a library of programming functions mainly aimed at computer vision.
#Very good for images and videos, especially real time videos.
#It is used extensively for facial recognition, object recognition, motion tracking,
#optical character recognition, segmentation, and even for artificial neural netwroks. 

You can import images in color, grey scale or unchanged usingindividual commands 
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

"""

import cv2

grey_img = cv2.imread("images/test_image.jpg", 0)
color_img = cv2.imread("images/test_image.jpg", 1)

#images opened using cv2 are numpy arrays
print(type(grey_img)) 
print(type(color_img)) 

# Use the function cv2.imshow() to display an image in a window. 
# First argument is the window name which is a string. second argument is our image. 

cv2.imshow("pic", grey_img)
cv2.imshow("color pic", color_img)

# Maintain output window until 
# user presses a key or 1000 ms (1s)
cv2.waitKey(0)          

#destroys all windows created
cv2.destroyAllWindows() 

#OpenCV imread, imwrite and imshow all work with the BGR order, not RGB
#but there is no need to change the order when you read an image with 
#cv2.imread and then want to show it with cv2.imshow
#if you use matplotlib, it uses RGB. 

import matplotlib.pyplot as plt
plt.imshow(color_img)  

#OpenCV represents RGB images as multi-dimensional NumPy arrays, but as BGR.

#we can convert the images from BGR to RGB
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

#We can also change color spaces from RGB to HSV..
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV))

"""
###############################################################################
###########Reading OME-TIFF using apeer_ometiff_library ###########
# pip install apeer-ometiff-library first 
# to import the package you need to use import apeer_ometiff_library
#OME-TIFF has tiff and metada (as XML) embedded
#Image is a 5D array.
"""

from apeer_ometiff_library import io  #Use apeer.com free platform for image processing in the cloud

(pic2, omexml) = io.read_ometiff("images/test_image.ome.tif")  #Unwrap image and embedded xml metadata
print (pic2.shape)   #to verify the shape of the array
print(pic2)

print(omexml)

"""
####################################################################################
#reading czi files
# pip install czifile 
# to import the package you need to use import czifile
# https://pypi.org/project/czifile/
"""

import czifile

img = czifile.imread('images/test_image.czi')
print(img.shape)


import czifile
from skimage import io

img = czifile.imread('images/Osteosarcoma_01.czi')
print(img.shape)
img1=img[0, 0, :, :, :, 0]
print(img1.shape)
img2=img1[2,:,:]
io.imshow(img2)

"""
######################################################################################
### Reading multiple images from a folder
#The glob module finds all the path names 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 
"""

#import the library opencv
import cv2
import glob

#select the path
path = "images/test_images/aeroplane/*.*"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    print(a)  #print numpy arrays for each file

#let us look at each file
#    cv2.imshow('Original Image', a)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    
#process each image - change color from BGR to RGB.
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    cv2.imshow('Color image', c)
#wait for 1 second
    k = cv2.waitKey(0)
#destroy the window
    cv2.destroyAllWindows()

#######################################################################################