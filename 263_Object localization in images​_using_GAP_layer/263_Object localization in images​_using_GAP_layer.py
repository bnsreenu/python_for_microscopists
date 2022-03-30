# https://youtu.be/GiyldmoYe_M
"""
Object localization in an image by leveraging the global average pool layer. 

Imagenet classes can be obtained from here:
    https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    OR
    https://github.com/Waikato/wekaDeeplearning4j/blob/master/docs/user-guide/class-maps/IMAGENET.md
"""

import numpy as np
import ast #to easily read out class text file that contains some unknwn syntax.
import scipy   #to upscale the image
import matplotlib.pyplot as plt
import cv2     
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model   
from PIL import Image

#Read an image (containing an object from one of the 1000 resnet50 classes.)
img = cv2.imread('images/dog_in_park.JPG', 1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img, 'RGB')
img = img.resize((224, 224))
img = np.array(img)
plt.imshow(img)

img_tensor = np.expand_dims(img, axis=0)
preprocessed_img = preprocess_input(img_tensor)

#Import the resnet50 model
model = ResNet50(weights='imagenet')
print(model.summary()) #Notice the Global Average Pooling layer at the last but one

#Get weights for the prediction layer (last layer)
#We should see 2048 weights for each of the 1000 classes (2048,1000)
last_layer_weights = model.layers[-1].get_weights()[0]  #Predictions layer


#Output both predictions (last layer) and conv5_block3_add (just before final activation layer)
ResNet_model = Model(inputs=model.input, 
        outputs=(model.layers[-4].output, model.layers[-1].output)) 

#Get the predictions and the output of last conv. layer. 
last_conv_output, pred_vec = ResNet_model.predict(preprocessed_img)
#Last conv. output for the image
last_conv_output = np.squeeze(last_conv_output) #7x7x2048
#Prediction for the image
pred = np.argmax(pred_vec)

# spline interpolation to resize each filtered image to size of original image 
h = int(img.shape[0]/last_conv_output.shape[0])
w = int(img.shape[1]/last_conv_output.shape[1])
upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1) # dim: 224 x 224 x 2048

#Get the weights from the last layer for the prediction class
last_layer_weights_for_pred = last_layer_weights[:, pred] # dim: (2048,) 

#To generate the final heat map. 
#Reshape the upsampled last conv. output to n x filters and multiply (dot product) 
# with the last layer weigths for the prediction. 
# Reshape back to the image size for easy overlay onto the original image. 
heat_map = np.dot(upsampled_last_conv_output.reshape((224*224, 2048)), 
                  last_layer_weights_for_pred).reshape(224,224) # dim: 224 x 224

#We can fetch the actual class name for the prediction so we can add it as
# a title to our plot.
with open('imagenet_classes.txt') as imagenet_classes_file:
        imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())
predicted_class = imagenet_classes_dict[pred]   

#Plot original image with heatmap overlaid. 
fig, ax = plt.subplots()
ax.imshow(img)
ax.imshow(heat_map, cmap='jet', alpha=0.5)
ax.set_title(predicted_class) 
plt.show()

