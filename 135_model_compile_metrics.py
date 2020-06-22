#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/rHgQrdME-DA

"""
Few lines of code to understand model.compile metrics

"""


#This part of the code demonstrates regression metrics in Keras
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt

# Define an input array. 
X = array([1,2,3,4,5,6,7,8,9,10])

# Define model
model = Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine'])

# train model
history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)

# plot metrics
plt.plot(history.history['mse'])
plt.plot(history.history['mae'])
plt.plot(history.history['mape'])
plt.plot(history.history['cosine'])
plt.show()


#This part of the code demonstrates classification accuracy metric in Keras
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from keras.utils import to_categorical

"""
####################################
#Understanding the artificial generated data
#Comment this part for full code execution.
# 
from sklearn.datasets import make_blobs
from matplotlib import pyplot
from numpy import where

# generate classification dataset with 3 centers (labels/classes)
X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
# scatter plot for each class value
for class_value in range(3):
	# select indices of points with the class label
	row_ix = where(y == class_value)
	# scatter plot for points with a different color
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show plot
plt.show()
"""
########################################


x, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)  
	# one hot encode output variable to convert from integers to binary class
y = to_categorical(y)

# create model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(3, activation='softmax'))  #Predict the 3 classes
    
	# compile model
	#opt = SGD(lr=0.01, momentum=0.9) #if we want to use Stochastic gradient descent as optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
history = model.fit(x, y, epochs=100, batch_size=32, verbose=2)
# plot metrics
plt.plot(history.history['accuracy'])
plt.show()