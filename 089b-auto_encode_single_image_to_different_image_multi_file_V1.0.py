#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://www.youtube.com/watch?v=Te3YieMUYd8


"""
@author: Sreenivas Bhattiprolu
Good example to demo image reconstruction using autoencoders
Try different optimizers and loss

To launch tensorboard type this in the console: !tensorboard --logdir=logs/ --host localhost --port 8088
then go to: http://localhost:8088/

"""

from matplotlib.pyplot import imshow
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
import os

SIZE=256


from tqdm import tqdm
img_data=[]
path1 = 'einstein_mona_lisa/einstein/'
files=os.listdir(path1)
for i in tqdm(files):
    img=cv2.imread(path1+'/'+i,1)   #Change 0 to 1 for color images
    img=cv2.resize(img,(SIZE, SIZE))
    img_data.append(img_to_array(img))
    

img2_data=[]
path2 = 'einstein_mona_lisa/monalisa/'
files=os.listdir(path2)
for i in tqdm(files):
    img=cv2.imread(path2+'/'+i,1)  #Change 0 to 1 for color images
    img=cv2.resize(img,(SIZE, SIZE))
    img2_data.append(img_to_array(img))

img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.

img_array2 = np.reshape(img2_data, (len(img2_data), SIZE, SIZE, 3))
img_array2 = img_array2.astype('float32') / 255.



#Original einstein image for prediction as monalisa
img_data3=[]

img3=cv2.imread('einstein_mona_lisa/einstein_original.jpg', 1)   #Change 0 to 1 for color images
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)#Changing BGR to RGB to show images in true colors
img3=cv2.resize(img3,(SIZE, SIZE))
img_data3.append(img_to_array(img3))

img_array3 = np.reshape(img_data3, (len(img_data3), SIZE, SIZE, 3))
img_array3 = img_array3.astype('float32') / 255.

import time
start=time.time()


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()

import tensorflow as tf
callbacks = [tf.keras.callbacks.TensorBoard(log_dir='einstein_logs')]


model.fit(img_array, img_array2,
        epochs=100000,
        shuffle=True,
        callbacks=callbacks)

finish=time.time()
print('total_time = ', finish-start)

model.save('einstein_autoencoder.model')

print("Output")
pred = model.predict(img_array3)



imshow(pred[0].reshape(SIZE,SIZE,3))
