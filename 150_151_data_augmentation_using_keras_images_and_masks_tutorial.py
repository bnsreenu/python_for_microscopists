#https://youtu.be/k4TqxHteJ7s
#https://youtu.be/mwN2GGA4mqo

"""
Data Augmentation for images and masks

Screws up labels for binary / multinary masks used for semantic segmentation 
as it interpolates pixels during transformation. 

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
        fill_mode='reflect')


x_img = io.imread('data/images/_Sandstone_Versa0000.png0_0.png')
x_mask = io.imread('data/masks/_Sandstone_Versa0000.tif.png0_0.png')


x_img = np.expand_dims(x_img, axis=0)
x_img = np.expand_dims(x_img, axis=3)

x_mask = np.expand_dims(x_mask, axis=0)
x_mask = np.expand_dims(x_mask, axis=3)

#Generate Images
i = 0
for batch in datagen.flow(x_img, batch_size=16,  
                          save_to_dir='data/augmented/augmented_images', 
                          save_prefix='aug_img', 
                          save_format='png',
                          seed=20):
    i += 1
    if i > 10:
        break  # otherwise the generator would loop indefinitely 

#Generate Images
j = 0
for batch in datagen.flow(x_mask, batch_size=16,  
                          save_to_dir='data/augmented/augmented_masks', 
                          save_prefix='aug_mask', 
                          save_format='tif',
                          seed=20):
    j += 1
    if j > 10:
        break  # otherwise the generator would loop indefinitely 











