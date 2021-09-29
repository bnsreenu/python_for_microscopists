# https://youtu.be/HXzz87WVm6c
"""
Code to convert keras h5 to tflite
Uncomment the converter.optimizations line if you want to optimize the tflite model
for edge devices. Please remember that optimized model will be very slow 
on Windows10. 

"""

#Code to convert h5 to tflite
import tensorflow as tf

model =tf.keras.models.load_model("models/malaria_model_100epochs.h5")

#Convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#Implement optimization strategy for smaller model sizes
#converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open("models/malaria_model_100epochs.tflite", "wb").write(tflite_model)



