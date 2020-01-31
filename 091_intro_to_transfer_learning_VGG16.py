#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/LOvrfvtiC8c


#

#Load the VGG model. For the first time it downloads weights from the Internet.
#Stored in Keras/Models directory. (Almost 600MB)
#We can include arguments to define whether we want to download full model, 
#Or part only, include weights, classes, etc. 

from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
model.summary()

#Let us load an image to test the pretrained VGG model.
#These models are developed on powerful computers so we may as well use them for transfer learning
#For VGG16 the images need to be 224x224. 
from keras.preprocessing.image import load_img
image = load_img('images/cab.jpg', target_size=(224, 224))

#Convert pixels to Numpy array                                        
from keras.preprocessing.image import img_to_array
image = img_to_array(image)

# Reshape data for the model. VGG expects multiple images of size 224x224x3, 
#therefore the input shape needs to be (1, 224, 224, 3)
#image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
import numpy as np
image = np.expand_dims(image, axis=0)

#Data needs to be preprocessed same way as the training dataset, to get best results
#preprocessing from Keras does this job. 
#Notice the change in pixel values (Preprocessing subtracts mean RGB value of training set from each pixel)
from keras.applications.vgg16 import preprocess_input
image = preprocess_input(image)


# predict the probability across all output categories.
#Probability for each of the 1000 classes will be calculated.
pred = model.predict(image)

#Print the probabilities of the top 5 classes
from tensorflow.keras.applications.mobilenet import decode_predictions
pred_classes = decode_predictions(pred, top=5)
for i in pred_classes[0]:
    print(i)