# https://youtu.be/QxzxLVzNfbI

"""
How to apply image processing operations to multiple images

"""

## Using GLOB
#Now, let us load images and perform some action.
#import the opencv library so we can use it to read and process images
import cv2
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte

#select the path
path = "test_images/imgs/*.*"
img_number = 1  #Start an iterator for image number.
#This number can be later added to output image file names.

for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path
    
#process each image - change color from BGR to RGB.
    smoothed_image = img_as_ubyte(gaussian(img, sigma=5, mode='constant', cval=0.0))
    
    cv2.imwrite("test_images/smoothed/smoothed_image"+str(img_number)+".jpg", smoothed_image)
    img_number +=1     

###########################################

#Using os library to walk through folders
import os
import cv2
from skimage.filters import gaussian
from skimage import img_as_ubyte

img_number = 1
for root, dirs, files in os.walk("test_images/imgs"):
#for path,subdir,files in os.walk("."):
#   for name in dirs:
#       print (os.path.join(root, name)) # will print path of directories
   for name in files:    
       print (os.path.join(root, name)) # will print path of files 
       path = os.path.join(root, name)
       img= cv2.imread(path, 0)  #now, we can read each file since we have the full path
       #process each image - change color from BGR to RGB.
       smoothed_image = img_as_ubyte(gaussian(img, sigma=5, mode='constant', cval=0.0))
       cv2.imwrite("test_images/smoothed/smoothed_image"+str(img_number)+".jpg", smoothed_image)
       img_number +=1     
       

################################################
#Capture all mages into an array and then iterate through each image
#Normally used for machine learning workflows.

import numpy as np
import cv2
import os
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte

images_list = []
SIZE = 512

path = "test_images/imgs/*.*"

#First create a stack array of all images
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img= cv2.imread(file, 0)  #now, we can read each file since we have the full path
    img = cv2.resize(img, (SIZE, SIZE))
    images_list.append(img)
        
images_list = np.array(images_list)

#Process each slice in the stack
img_number = 1
for image in range(images_list.shape[0]):
    input_img = images_list[image,:,:]  #Grey images. For color add another dim.
    smoothed_image = img_as_ubyte(gaussian(input_img, sigma=5, mode='constant', cval=0.0))
    cv2.imwrite("test_images/smoothed/smoothed_image"+str(img_number)+".jpg", smoothed_image)
    img_number +=1     
       
########################################################
#Reading multidimensional tif images and processing slice by slice
    
import numpy as np
import cv2
import os
import glob
from skimage.filters import gaussian
from skimage import img_as_ubyte

# file = 'test_images/scratch_time_series.tif'
# img= cv2.imread(file, 0)     

import tifffile
img = tifffile.imread(file)    

img_number = 1
for image in range(img.shape[0]):
    input_img = img[image,:,:]  #Grey images. For color add another dim.
    smoothed_image = img_as_ubyte(gaussian(input_img, sigma=5, mode='constant', cval=0.0))
    cv2.imwrite("test_images/smoothed/smoothed_image"+str(img_number)+".jpg", smoothed_image)
    img_number +=1     
   
    
    