# https://youtu.be/65qPtD6khzg
"""
Create border pixels from binary masks. 
We can include these border pixels as another class to train a multiclass semantic segmenter
What is the advantage?
We can use border pixels to perform watershed and achieve 'instance' segmentation. 

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#Function to define border. 
#Just erode some pixels into objects and dilate tooutside the objects. 
#This region would be the border. Replace border pixel value to something other than 255. 
def generate_border(image, border_size=5, n_erosions=1):

    erosion_kernel = np.ones((3,3), np.uint8)      ## Start by eroding edge pixels
    eroded_image = cv2.erode(image, erosion_kernel, iterations=n_erosions)  
 
    ## Define the kernel size for dilation based on the desired border size (Add 1 to keep it odd)
    kernel_size = 2*border_size + 1 
    dilation_kernel = np.ones((kernel_size, kernel_size), np.uint8)   #Kernel to be used for dilation
    dilated  = cv2.dilate(eroded_image, dilation_kernel, iterations = 1)
    #plt.imshow(dilated, cmap='gray')
    
    ## Replace 255 values to 127 for all pixels. Eventually we will only define border pixels with this value
    dilated_127 = np.where(dilated == 255, 127, dilated) 	
    
    #In the above dilated image, convert the eroded object parts to pixel value 255
    #What's remaining with a value of 127 would be the boundary pixels. 
    original_with_border = np.where(eroded_image > 127, 255, dilated_127)
    
    #plt.imshow(original_with_border,cmap='gray')
    
    return original_with_border
 

#select the path
from pathlib import Path
path = "data/generated_patches/masks/*.*"

for file in glob.glob(path):
    name = Path(file).stem  #Extract name so processed images can be saved under the same name
    img= cv2.imread(file, cv2.IMREAD_GRAYSCALE)  #now, we can read each file since we have the full path    
    #process each image
    processed_image = generate_border(img, border_size=5, n_erosions=1)
    #Save images with same name as original image/mask
    cv2.imwrite("data/generated_patches/masks_with_border/"+ name + ".tif", processed_image)
    print("Finished processing image ", name)     
   
    
    
    
    
    
    
    