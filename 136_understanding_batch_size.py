#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/Y-zswp6Yxf0

"""
Effect of Batch Size on Model Behavior

Parts of code adapted from 'Machinelearningmastery.com blog posts.
"""

from sklearn.datasets import make_blobs  #To generate artificial data
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot as plt

"""
####################################
#Understanding the artificial generated data.
#Plot to see the blobs.
#Comment this part during full code execution.
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

############################################

# prepare train and test dataset
def prepare_data():
	#  generate classification dataset with 3 centers (labels/classes)
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=2)
    
	# one hot encode output variable to convert from integers to binary class
	y = to_categorical(y)
    
	# split into train and test
	n_train = 500
	X_train, X_test = X[:n_train, :], X[n_train:, :]
	y_train, y_test = y[:n_train], y[n_train:]
	return X_train, y_train, X_test, y_test

# fit a model and plot learning curve
def fit_model(X_train, y_train, X_test, y_test, n_batch):
	# define model
	model = Sequential()
	model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))  #Predict the 3 classes
    
	# compile model
	#opt = SGD(lr=0.01, momentum=0.9) #if we want to use Stochastic gradient descent as optimizer
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
	# fit model
	history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=1, batch_size=n_batch)
	# plot learning curves
	plt.plot(history.history['accuracy'], label='train')
	plt.plot(history.history['val_accuracy'], label='test')
	plt.title('batch='+str(n_batch), pad=-40)

# prepare dataset
X_train, y_train, X_test, y_test = prepare_data()

# create learning curves for different batch sizes
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 450]
plt.figure(figsize=(16,10))
for i in range(len(batch_sizes)):
	# determine the plot number
	plot_no = 420 + (i+1)   #Plot 4x2 layout
	plt.subplot(plot_no)
	# fit model and plot learning curves for a batch size
	fit_model(X_train, y_train, X_test, y_test, batch_sizes[i])
# show learning curves
plt.show()