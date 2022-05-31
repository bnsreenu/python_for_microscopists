
"""
@author: Sreenivas Bhattiprolu

Image shifts via the width_shift_range and height_shift_range arguments.
Image flips via the horizontal_flip and vertical_flip arguments.
Image rotations via the rotation_range argument
Image brightness via the brightness_range argument.
Image zoom via the zoom_range argument.
"""

from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np

# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor. 

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect')    #Also try nearest, constant, reflect, wrap


######################################################################
#Loading a single image for demonstration purposes.
#Using flow method to augment the image

# Loading a sample image  
#Can use any library to read images but they need to be in an array form
#If using keras load_img convert it to an array first
x_img = io.imread('calgary_images/images/test_256px.tif')  #Array with shape (256, 256, 3)
x_mask = io.imread('calgary_images/masks/test_256px.tif_annotation.tif')
# Reshape the input image because ...
#x: Input data to datagen.flow must be Numpy array of rank 4 or a tuple.
#First element represents the number of images
x_img = np.expand_dims(x_img, axis=0)
#x_img = np.expand_dims(x_img, axis=3)

x_mask = np.expand_dims(x_mask, axis=0)
#x_mask = np.expand_dims(x_mask, axis=3)


#Generate Images
i = 0
for batch in datagen.flow(x_img, batch_size=16,  
                          save_to_dir='augmented_images/images', 
                          save_prefix='aug_img', 
                          save_format='tif',
                          seed=99):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely  

#Generate Masks
j = 0
for batch in datagen.flow(x_mask, batch_size=16,  
                          save_to_dir='augmented_images/masks', 
                          save_prefix='aug_mask', 
                          save_format='tif',
                          seed=99):
    j += 1
    if j > 20:
        break 
####################################################################
