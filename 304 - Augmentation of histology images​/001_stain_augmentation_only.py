# https://youtu.be/SuDtHqtC5OE

import PIL.Image as Image
import os
from torchvision import transforms as transforms

dir_path = 'data/original/'
img_list = os.listdir(dir_path)

save_dir_path = "data/stain_augmentation"
if not os.path.exists(save_dir_path):
    os.mkdir(save_dir_path)


#ColorJitter randomly changes the brightness, contrast, saturation and hue of an image.
    
for img_path in img_list:
    full_img_path = dir_path+img_path
    image = transforms.ColorJitter(
        brightness=0.35, contrast=0.5, saturation=0.5, hue=0.5
    )(Image.open(full_img_path))
    save_img_path = save_dir_path + "/{}".format(img_path.split("/")[-1])
    image.save(save_img_path)



"""
For TensorFlow based transformations, you can use...

import tensorflow as tf
brightness = tf.image.RandomBrightness
contrast = tf.image.RandomContrast
tf.image.random_saturation
tf.image.random_hue

"""