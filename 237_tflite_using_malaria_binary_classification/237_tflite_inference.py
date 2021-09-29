# https://youtu.be/HXzz87WVm6c
"""
Inference using tflite and comparing against h5
"""

import numpy as np
import tensorflow as tf

from keras.preprocessing.image import load_img, img_to_array
from time import time
from matplotlib import pyplot as plt
import os

#Parasitized image
#image = load_img('cell_images/Parasitized/C39P4thinF_original_IMG_20150622_111206_cell_99.png', target_size=(150,150))

#Uninfected image
image = load_img('cell_images/Uninfected/C12NThinF_IMG_20150614_124212_cell_161.png', target_size=(150,150))

image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print(image.shape)

from keras.utils import normalize
image = normalize(image, axis=1)

###############################################
###PREDICT USING REGULAR KERAS TRAINED MODEL FILE (h5). 
##########################################################
keras_model_size = os.path.getsize("models/malaria_model_100epochs.h5")/1048576  #Convert to MB
print("Keras Model size is: ", keras_model_size, "MB")
#Using regular keral model
model = tf.keras.models.load_model("models/malaria_model_100epochs.h5")

time_before=time()
keras_prediction = model.predict(image)
time_after=time()
total_keras_time = time_after - time_before
print("Total prediction time for keras model is: ", total_keras_time)

print("The keras prediction for this image is: ", keras_prediction, " 0=Uninfected, 1=Parasited")



##################################################################################
#### PREDICT USING tflite ###
############################################################################
tflite_size = os.path.getsize("models/malaria_model_100epochs.tflite")/1048576  #Convert to MB
print("tflite Model without opt. size is: ", tflite_size, "MB")
#Not optimized (file size = 540MB). Taking about 0.5 seconds for inference
tflite_model_path = "models/malaria_model_100epochs.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Load image
input_data = image

interpreter.set_tensor(input_details[0]['index'], input_data)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_time = time_after - time_before
print("Total prediction time for tflite without opt model is: ", total_tflite_time)

output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")

#################################################################
#### PREDICT USING tflite with optimization###
#################################################################
tflite_optimized_size = os.path.getsize("models/malaria_model_100epochs_optimized.tflite")/1048576  #Convert to MB
print("tflite Model with optimization size is: ", tflite_optimized_size, "MB")
#Optimized using default optimization strategy (file size = 135MB). Taking about 39 seconds for prediction
tflite_optimized_model_path = "models/malaria_model_100epochs_optimized.tflite"


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_optimized_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Load image
input_data = image

interpreter.set_tensor(input_details[0]['index'], input_data)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_opt_time = time_after - time_before
print("Total prediction time for tflite model with opt is: ", total_tflite_opt_time)

output_data_tflite_opt = interpreter.get_tensor(output_details[0]['index'])
print("The tflite with opt prediction for this image is: ", output_data_tflite_opt, " 0=Uninfected, 1=Parasited")

#############################################

#Summary
print("###############################################")
print("Keras Model size is: ", keras_model_size)
print("tflite Model without opt. size is: ", tflite_size)
print("tflite Model with optimization size is: ", tflite_optimized_size)
print("________________________________________________")
print("Total prediction time for keras model is: ", total_keras_time)
print("Total prediction time for tflite without opt model is: ", total_tflite_time)
print("Total prediction time for tflite model with opt is: ", total_tflite_opt_time)
print("________________________________________________")
print("The keras prediction for this image is: ", keras_prediction, " 0=Uninfected, 1=Parasited")
print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")
print("The tflite with opt prediction for this image is: ", output_data_tflite_opt, " 0=Uninfected, 1=Parasited")


