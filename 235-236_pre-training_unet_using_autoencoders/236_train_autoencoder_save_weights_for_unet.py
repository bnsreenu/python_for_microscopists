# https://youtu.be/hTpq9lzAb8M
"""
@author: Sreenivas Bhattiprolu

Train an autoencoder for a given type of images. e.g. EM images of cells.
Use the encoder part of the trained autoencoder as the encoder for a U-net.
Use pre-trained weights from autoencoder as starting weights for encoder in the Unet. 
Train the Unet.

Training with initial encoder pre-trained weights would dramatically speed up 
the training process of U-net. 

"""

import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential
import os
from keras.models import Model
from matplotlib import pyplot as plt
SIZE=256


from tqdm import tqdm
img_data=[]
path1 = 'data/mito/images/'
files=os.listdir(path1)
for i in tqdm(files):
    img=cv2.imread(path1+'/'+i,1)   #Change 0 to 1 for color images
    img=cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))
    

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

#In the interest of time let us train on 500 images
img_array2 = img_array[200:700]
#
##########################################
#Define the autoencoder model. 
#Experiment with various optimizers and loss functions
from models import build_autoencoder, build_encoder, build_unet

autoencoder_model=build_autoencoder(img.shape)
autoencoder_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(autoencoder_model.summary())

############################
#Train the autoencoder

history = autoencoder_model.fit(img_array2, img_array2,
        epochs=100, verbose=1)


autoencoder_model.save('autoencoder_mito_500imgs_100epochs.h5')

#test on a few images
#Load the model 
from keras.models import load_model
autoencoder_model = load_model("autoencoder_mito_500imgs_100epochs.h5", compile=False)
       
import random
num=random.randint(0, len(img_array2)-1)
test_img = np.expand_dims(img_array[num], axis=0)
pred = autoencoder_model.predict(test_img)

plt.subplot(1,2,1)
plt.imshow(test_img[0])
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(pred[0].reshape(SIZE,SIZE,3))
plt.title('Reconstructed')
plt.show()


########################################################
###############################

#Extract weights only for the encoder part of the Autoencoder. 

#from models import build_autoencoder
#from keras.models import load_model
autoencoder_model = load_model("autoencoder_mito_500imgs_100epochs.h5", compile=False)
       
#Now define encoder model only, without the decoder part. 
input_shape = (256, 256, 3)
input_img = Input(shape=input_shape)

encoder = build_encoder(input_img)
encoder_model = Model(input_img, encoder)
print(encoder_model.summary())

num_encoder_layers = len(encoder_model.layers) #35 layers in our encoder. 

#Get weights for the 35 layers from trained autoencoder model and assign to our new encoder model 
for l1, l2 in zip(encoder_model.layers[:35], autoencoder_model.layers[0:35]):
    l1.set_weights(l2.get_weights())

#Verify if the weights are the same between autoencoder and encoder only models. 
autoencoder_weights = autoencoder_model.get_weights()[0][1]
encoder_weights = encoder_model.get_weights()[0][1]

#Save encoder weights for future comparison
np.save('pretrained_encoder-weights.npy', encoder_weights )


#Check the output of encoder_model on a test image
#Should be of size 16x16x1024 for our model
temp_img = cv2.imread('data/mito/images/img9.tif',1)
temp_img = temp_img.astype('float32') / 255.
temp_img = np.expand_dims(temp_img, axis=0)
temp_img_encoded=encoder_model.predict(temp_img)

#Plot a few encoded channels

################################################
#Now let us define a Unet with same encoder part as out autoencoder. 
#Then load weights from the original autoencoder for the first 35 layers (encoder)
input_shape = (256, 256, 3)
unet_model = build_unet(input_shape)

#Print layer names for each model to verify the layers....
#First 35 layers should be the same in both models. 
unet_layer_names=[]
for layer in unet_model.layers:
    unet_layer_names.append(layer.name)

autoencoder_layer_names = []
for layer in autoencoder_model.layers:
    autoencoder_layer_names.append(layer.name)
    
#Make sure the first 35 layers are the same. Remember that the exct names of the layers will be different.
###########

#Set weights to encoder part of the U-net (first 35 layers)
for l1, l2 in zip(unet_model.layers[:35], autoencoder_model.layers[0:35]):
    l1.set_weights(l2.get_weights())

from keras.optimizers import Adam
import segmentation_models as sm
unet_model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
#unet_model.compile(optimizer=Adam(lr = 1e-3), loss='binary_crossentropy', metrics=['accuracy'])
unet_model.summary()
print(unet_model.output_shape)

unet_model.save('unet_model_weights.h5')


#Now use train_unet to load these weights for encoder and train a unet model. 
###################################################################
