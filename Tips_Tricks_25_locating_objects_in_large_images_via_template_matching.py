# https://youtu.be/cDP_4VbC_sE
"""
Locating objects in large images using template matching

Need a source image and a template image.
The template image T is slided over the source image (as in 2D convolution), 
and the program tries to find matches using statistics.
Several comparison methods are implemented in OpenCV.
It returns a grayscale image, where each pixel denotes how much does the 
neighbourhood of that pixel match with template.

Once you get the result, you can use cv2.minMaxLoc() function 
to find where is the maximum/minimum value. 
Take it as the top-left corner of the rectangle and take (w,h) as 
width and height of the rectangle. 
That rectangle can be drawn on the region of matched template.

If the template image is larger than its size in the large image, we can perfrom
the same exercise by resizing the template image to multiple sizes. 
We can then extract the match with best score. 
"""
### Template matching, single object in an image.


import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('carl_zeiss_team.jpg')  #Large image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('sreeni40x40.jpg', 0)  #Small image (template)
h, w = template.shape[::] 

#methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.
plt.imshow(res, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img_rgb, top_left, bottom_right, (0, 0, 255), 2)  #Red rectangle with thickness 2. 

cv2.imwrite('matched.jpg', img_rgb)

       
### Template matching - multiple size checks
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('carl_zeiss_team.jpg')  #Large image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('sreeni.jpg', 0)  #Small image (template)


import imutils
# Clculate the metric for varying image sizes
#pick the one that gives the best metric (e.g. Minimum Sq Diff.)

best_match = None
for scale in np.linspace(0.055, 0.5, 11):  #Pick scale based on your estimate of template to object in the image ratio
    print(scale)

    
#Resize the input template image
    resized_template = imutils.resize(template, width = int(template.shape[1] * scale))
    
    res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_SQDIFF)
    min_val, _, min_loc, _ = cv2.minMaxLoc(res)  #Only care about minimum value and location as we are using TM_SQDIFF
      
    #Check if the min_val is the minimum compared to the value from other scales templates
    #If it is minimum then we got a better match compared to other scales
    #So save the value and location. 
    if best_match is None or min_val <= best_match[0]:
        ideal_scale=scale  #Save the ideal scale for printout. 
        h, w = resized_template.shape[::] #Get the size of the scaled template to draw the rectangle. 
        best_match = [min_val, min_loc, ideal_scale]
        
        
print("Ideal template image size is : ", int(template.shape[0]*ideal_scale), "x", int(template.shape[1]*ideal_scale))

#Save the image with a red box around the detected object in the large image. 
top_left = best_match[1]  #Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_rgb, top_left, bottom_right, (0, 0,255), 2)  #Red rectangle with thickness 2. 
cv2.imwrite('matched_resized.jpg', img_rgb)


