# https://youtu.be/drcagR2zNpw

"""
@author: Sreenivas Bhattiprolu

LearningRateScheduler
At the beginning of every epoch, this callback gets the updated learning rate value

Exponential learning rate: Instead of constant steps for learning rate 
a decreasing exponential function is used as epochs go by. This is not adaptive learning.

Adam is an Adaptive gradient descent algorithm, alternative to SGD where we have
static learning rate or pre-define the way learning rate updates. 
"""

import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
from keras.optimizers import SGD

from keras.callbacks import LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt

print(tf.__version__)
print(keras.__version__)

# fix random seed for reproducibility
np.random.seed(42)

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])


# normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# reshape images to 1D so we can just work with dense layers
#For this demo purposes
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

num_classes = 10

# One hot encoding for categorical labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

epochs=50
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

#Default values for SGD. lr=0.1, m=0, decay=0
#Nesterov momentum is a different version of the momentum method.
#Nesterov has stronger theoretical converge guarantees for convex functions.
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

# build the model
input_dim = x_train.shape[1]

def define_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform', 
                    input_dim = input_dim)) 
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer='he_uniform', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='he_uniform', activation='softmax'))
    
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])
    return model


model = define_model()
print(model.summary())

# Fit the model
batch_size = 64  #Try 100

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))


#######################################################################
#CUSTOM LEARNING RATE
#Time based decay: lr *= (1. / (1. + decay * iterations))

#Step decay drops the learning rate by a factor every few epochs.: 
# lr = lr0 * drop^floor(epoch / epochs_drop) 
#import math
#def step_decay(epoch):
#   initial_lrate = 0.1
#   drop = 0.5
#   epochs_drop = 10.0
#   lrate = initial_lrate * math.pow(drop,  
#           math.floor((1+epoch)/epochs_drop))
#   return lrate


#Eexponential decay
#To be used instead of static learning rate of 0.1
# 𝑙𝑟=𝑙𝑟₀ × 𝑒**(−𝑘𝑡)
#lr can still be used as initial value. 


#NOTE: Redefine model with a new name to make sure we start from scratch

exp_model = define_model()
print(exp_model.summary())


def exp_decay(epoch):
    lrate = learning_rate * np.exp(-decay_rate*epoch)
    return lrate

# learning schedule callback
lr_rate = LearningRateScheduler(exp_decay)
callbacks_list = [lr_rate]

exp_history = exp_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=callbacks_list,
                        verbose=1,
                        validation_data=(x_test, y_test))


############################################################################
#PLot for the model with constant learning rate
# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(history.history['loss']), 'r', label='train')
ax.plot(np.sqrt(history.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(history.history['acc']), 'r', label='train')
ax.plot(np.sqrt(history.history['val_acc']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

#####################################################
#Plot for the second model with exponential decay
# Plot the loss function
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(exp_history.history['loss']), 'r', label='train')
ax.plot(np.sqrt(exp_history.history['val_loss']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(exp_history.history['acc']), 'r', label='train')
ax.plot(np.sqrt(exp_history.history['val_acc']), 'b' ,label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)