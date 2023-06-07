# https://youtu.be/_GAhbrGHaVo
"""
Reinhard color transfer 
Based on the paper: https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

This approach is suitable for stain normalization of pathology images where
the 'look and feel' of all images can be normalized to a template image. 
This can be a good preprocessing step for machine learning and deep learning 
of pathology images. 

"""

import numpy as np
import cv2
import os

#input_dir = "input_images/"
input_dir = "pathology_input/"
input_image_list = os.listdir(input_dir)

#output_dir = "output_images/"
output_dir = "pathology_output/"

def get_mean_and_std(x):
	x_mean, x_std = cv2.meanStdDev(x)
	x_mean = np.hstack(np.around(x_mean,2))
	x_std = np.hstack(np.around(x_std,2))
	return x_mean, x_std

#template_img = cv2.imread('template_images/sunset_template.jpg')
template_img = cv2.imread('pathology_template/3.png')
template_img = cv2.cvtColor(template_img,cv2.COLOR_BGR2LAB)
template_mean, template_std = get_mean_and_std(template_img)

for img in (input_image_list):
    print(img)
    input_img = cv2.imread(input_dir+img)
    input_img = cv2.cvtColor(input_img,cv2.COLOR_BGR2LAB)
    
    
    img_mean, img_std = get_mean_and_std(input_img)
    
    
    height, width, channel = input_img.shape
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
            	x = input_img[i,j,k]
            	x = ((x-img_mean[k])*(template_std[k]/img_std[k]))+template_mean[k]
            	x = round(x)
            	# boundary check
            	x = 0 if x<0 else x
            	x = 255 if x>255 else x
            	input_img[i,j,k] = x
            
    input_img= cv2.cvtColor(input_img,cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_dir+"modified_"+img, input_img)

