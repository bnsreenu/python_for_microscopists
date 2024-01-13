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

from tensorflow.keras.applications.resnet50  import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

#Understand the ResNet50 model.

# # Get ResNet-50 Model
# def ResNet50Model_Demo(lastFourTrainable=True):
#   input_shape = (224, 224, 3)
#   resnet_model_demo = ResNet50(weights='imagenet', input_shape=input_shape, include_top=True)
#   output = resnet_model_demo.get_layer('avg_pool').output
#   resnet_model_demo = Model(resnet_model_demo.input, output)
#   resnet_model_demo.summary()
  
#   return resnet_model_demo
# ResNet50Model_Demo()


# #See how the final output gives us a vector of size 2048


class getResNet50Model:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.resnet_model = ResNet50(weights='imagenet', input_shape=self.input_shape, include_top = True)
        self.output = self.resnet_model.get_layer('avg_pool').output
        self.resnet_model = Model(self.resnet_model.input, self.output)
        #self.resnet_model.summary()
        

    '''
    Use Resnet50 model to extract features
    Output normalized feature vector
    '''
    def extract_feat(self, img_path):
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.resnet_model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat





