#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=e69JGAtA5gA


"""
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html

This code performs grain size distribution analysis and dumps results into a csv file.
It uses watershed segmentation for better segmentation.
Compare results to regular segmentation. 
For multi file, so we add a bunch of for loops to cycle through images. 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io


import glob
#The glob module finds all the pathnames 
#matching a specified pattern according to the rules used by the Unix shell
#The glob.glob returns the list of files with their full path 

pixels_to_um = 0.5 # 1 pixel = 500 nm (got this from the metadata of original image)
propList = ['Area',
                'equivalent_diameter', #Added... verify if it works
                'orientation', #Added, verify if it works. Angle btwn x-axis and major axis.
                'MajorAxisLength',
                'MinorAxisLength',
                'Perimeter',
                'MinIntensity',
                'MeanIntensity',
                'MaxIntensity']  
output_file = open('images/grains/grain_measurements.csv', 'w')
output_file.write('FileName' + "," + 'Grain #'+ "," + "," + ",".join(propList) + '\n') 
#join strings in array by commas. First column file name and 2nd column cell number 
#then join all names in prop list with a comma before their name. The go to next line.
#This will be the first row in the cs file. 
#Other rows will be filled by data generated later. 

#select the path
path = "images/grains/*.jpg"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    img1= cv2.imread(file)  #now, we can read each file since we have the full path

    img = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)

#Threshold image to binary using OTSU. ALl thresholded pixels will be set to 255
    ret1, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphological operations to remove small noise - opening
#To remove holes we can use closing
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)


#Now we know that the regions at the center of cells is for sure cells
#The region far away is background.
#We need to extract sure regions. For that we can use erode. 
#But we have cells touching, so erode alone will not work. 
#To separate touching objects, the best approach would be distance transform and then thresholding.

# let us start by identifying sure background area
# dilating pixes a few times increases cell boundary to background. 
# This way whatever is remaining for sure will be background. 
#The area in between sure background and foreground is our ambiguous area. 
#Watershed should find this area for us. 
    sure_bg = cv2.dilate(opening,kernel,iterations=2)


# Finding sure foreground area using distance transform and thresholding
#intensities of the points inside the foreground regions are changed to 
#distance their respective distances from the closest 0 value (boundary).
#https://www.tutorialspoint.com/opencv/opencv_distance_transformation.htm
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)


#Let us threshold the dist transform by starting at 1/2 its max value.
#print(dist_transform.max()) gives about 21.9
    ret2, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

#0.2* max value seems to separate the cells well.
#High value like 0.5 will not recognize some grain boundaries.

# Unknown ambiguous region is nothing but bkground - foreground
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

#Now we create a marker and label the regions inside. 
# For sure regions, both foreground and background will be labeled with positive numbers.
# Unknown regions will be labeled 0. 
#For markers let us use ConnectedComponents. 
    ret3, markers = cv2.connectedComponents(sure_fg)

#One problem rightnow is that the entire background pixels is given value 0.
#This means watershed considers this region as unknown.
#So let us add 1 to all labels so that sure background is not 0, but 1
    markers = markers+10

# Now, mark the region of unknown with zero
    markers[unknown==255] = 0
#plt.imshow(markers)   #Look at the 3 distinct regions.

#Now we are ready for watershed filling. 
    markers = cv2.watershed(img1,markers)
#The boundary region will be marked -1
#https://docs.opencv.org/3.3.1/d7/d1b/group__imgproc__misc.html#ga3267243e4d3f95165d55a618c65ac6e1

#Let us color boundaries in yellow. 
    img1[markers == -1] = [0,255,255]  

    img2 = color.label2rgb(markers, bg_label=0)

    cv2.imshow('Overlay on original image', img1)
    cv2.imshow('Colored Grains', img2)
    cv2.waitKey(0)

#Now, time to extract properties of detected cells
# regionprops function in skimage measure module calculates useful parameters for each object.
    regions = measure.regionprops(markers, intensity_image=img)

#Can print various parameters for all objects
    for prop in regions:
        print('Label: {} Area: {}'.format(prop.label, prop.area))

#Best way is to output all properties to a csv file
#Let us pick which ones we want to export. 

    grain_number = 1
    for region_props in regions:
        output_file.write(file+",")
        output_file.write(str(grain_number) + ',')
    #output cluster properties to the excel file
#        output_file.write(str(region_props['Label']))
        for i,prop in enumerate(propList):
            if(prop == 'Area'): 
                to_print = region_props[prop]*pixels_to_um**2   #Convert pixel square to um square
            elif(prop == 'orientation'): 
                to_print = region_props[prop]*57.2958  #Convert to degrees from radians
            elif(prop.find('Intensity') < 0):          # Any prop without Intensity in its name
                to_print = region_props[prop]*pixels_to_um
            else: 
                to_print = region_props[prop]     #Reamining props, basically the ones with Intensity in its name
            output_file.write(',' + str(to_print))
        output_file.write('\n')
        grain_number += 1

output_file.close()   #Closes the file, otherwise it would be read only. 