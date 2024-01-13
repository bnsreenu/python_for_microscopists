# https://youtu.be/zN9ZINn7g24
"""
@author: DigitalSreeni

This file provides the VGG16 method to be applied on images to extract features. 
These features can then be used for content based image retrieval. 
Please note that Imagenet pre-trained weights will be loaded to VGG16
and the output from the final layer has a shape of 512 - our feature vector length. 


"""

import numpy as np
from numpy import linalg as LA

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

#Understand the VGG16 model.
# model = VGG16(weights = 'imagenet', 
#               input_shape = ((224, 224, 3)), 
#               pooling = 'max', 
#               include_top = False)
# model.summary()
#See how the final output gives us a vector oof size 512


class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model.predict(np.zeros((1, 224, 224 , 3)))
        

    '''
    Use vgg16 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat

