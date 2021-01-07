# https://youtu.be/pU67vp4T8J4
"""
@author: Sreenivas Bhattiprolu

Grid search hyperparameters - Tuning activation functions

ONly trick is to define default activation.
Otherwise it throws error due tonot having all required parameters.

Use GridSearchCV class from scikit-learn

Keras models can be used in scikit-learn by wrapping them with 
the KerasClassifier or KerasRegressor class.

In GridSearchCV:
n_jobs = -1 --> uses multiple cores (parallelized)
n_jobs = 1 --> do not parallelize (do this only if you get an error with -1)


"""

import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout

from keras.callbacks import LearningRateScheduler

import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import GridSearchCV

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

#Take a subset of train for grid search. Let us take 10% for now
from sklearn.model_selection import train_test_split
x_grid, x_not_use, y_grid, y_not_use = train_test_split(x_train, y_train, test_size=0.9, random_state=42)

# build the model
input_dim = x_grid.shape[1]


#NOTE: Add default optimizer, otherwise throws error 'optimizer not legal parameter'
def define_model(activation='relu', init_weights='uniform', optimizer='Adam'):   
    model = Sequential()
    model.add(Dense(64, activation=activation, kernel_initializer=init_weights, 
                    input_dim = input_dim)) 
    model.add(Dropout(0.1))
    model.add(Dense(64, kernel_initializer=init_weights, activation=activation))
    model.add(Dense(num_classes, kernel_initializer=init_weights, activation='softmax'))
    
    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,      
                  metrics=['acc'])
    return model

# implement the Scikit-Learn classifier interface
# requires model defined as a function, which we already have
from keras.wrappers.scikit_learn import KerasClassifier
batch_size = 100
epochs = 10

model = KerasClassifier(build_fn=define_model, 
                        epochs=epochs, 
                        batch_size = batch_size, 
                        verbose=1)

activation = ['softmax', 'relu', 'sigmoid']
#Also try softplus, tanh, linear, hard_sigmoid 
init_weights = ['uniform', 'normal', 'he_uniform']
#Also try lecun_uniform, he_normal, glorot_normal, etc. 
optimizer = ['SGD', 'RMSprop', 'Adam']

param_grid = dict(activation=activation, init_weights=init_weights, optimizer=optimizer)

#n_jobs=16 uses 16 CPUs. Try not to do -1 on your system as it may hang!!!
# -1 refers to using all available CPUs
#Cross validation, cv=3
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=16, cv=3)

grid_result = grid.fit(x_grid, y_grid)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("Mean = %f (std=%f) with: %r" % (mean, stdev, param))


