# https://youtu.be/gNRVTCf6lvY
"""
What is the Global Average Pooling (GAP layer)
and how it can be used to summrize features in an image

"""

from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, AveragePooling2D
from tensorflow.keras.models import Sequential, Model

import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#Create a 4x4 dummy image for demonstration
my_img = np.array([[4.,7,3,1], 
                   [1,5,2,3], 
                   [6,3,9,2], 
                   [1,6,3,7]])

#Expand dims to make it ready for layer input (4 dim)
input_img = my_img[np.newaxis,:,:,np.newaxis]

# Max pooling: For each pool, the operation computes the max value.
#Commonly used in deep learning to downsample an input. 
#Commonly used because we assume that our objects of interest likely produces
#large values, so maxpooling is commonly used compared to average or other pooling. 
max_pool = MaxPooling2D(pool_size=(2, 2))(input_img)
max_pool=max_pool.numpy()
print(max_pool.shape)
print(max_pool)

# Average pooling: For each pool, the operation computes the average value.
# This can be useful in situations where average information is important. 
#In general, you get similar results to maxpooling. 
avg_pool = AveragePooling2D(pool_size=(2, 2))(input_img)
avg_pool=avg_pool.numpy()
print(avg_pool.shape)
print(avg_pool)

#Global average pooling: Average of entire image is calculated instead of a smaller block. 
#Instead of down sampling patches of the input feature map, global pooling down samples the entire feature map to a single value.
#These operations can be useful many way, especially when interested in localized information.
#They are used commonly to replace flattening and dense layers. 
#Models typically end with convolotuion layers upon which GAP layer will be executed. 
#GAP converts each feature map into a single number. 
#Since feature maps learn about objects in the image, the GAP output can be used
#to map the presence of objects for various classes in images. 
#The GAP output is connected to softmax to obtain the multiclass probability distribution. 
GAP = GlobalAveragePooling2D()(input_img)
GAP=GAP.numpy()
print(GAP.shape)
print(GAP)

print(my_img.mean()) #In this case we have a single image.

################################################################################

#Quick overview of GAP to highlight the presence of features in an image. 


image_directory = 'cell_images/'
SIZE = 256
dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
label = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

parasitized_images = os.listdir(image_directory + 'Parasitized/')
for i, image_name in enumerate(parasitized_images):    #Remember enumerate method adds a counter and returns the enumerate object
    
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Parasitized/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(1)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

uninfected_images = os.listdir(image_directory + 'Uninfected/')
for i, image_name in enumerate(uninfected_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + 'Uninfected/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
        label.append(0)

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.20, random_state = 0)

X_train = X_train /255.
X_test = X_test /255.

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)

avg_pool_model = Sequential()

avg_pool_model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
avg_pool_model.add(Activation('relu'))
avg_pool_model.add(MaxPooling2D(pool_size=(2, 2)))

avg_pool_model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
avg_pool_model.add(Activation('relu'))
avg_pool_model.add(MaxPooling2D(pool_size=(2, 2)))

avg_pool_model.add(Conv2D(64, (3, 3), kernel_initializer = 'he_uniform'))

avg_pool_model.add(GlobalAveragePooling2D())

#avg_pool_model.add(Flatten()) #No need for flattening anymore.
avg_pool_model.add(Dense(1))
avg_pool_model.add(Activation('sigmoid'))  

avg_pool_model.compile(loss='binary_crossentropy',
              optimizer='adam',             
              metrics=['accuracy'])

print(avg_pool_model.summary())

history_avg_pool = avg_pool_model.fit(X_train, 
                         y_train, 
                         batch_size = 64, 
                         verbose = 1, 
                         epochs = 30,      
                         validation_data=(X_test,y_test),
                         shuffle = False
                     )



#plot the training and validation accuracy and loss at each epoch
loss = history_avg_pool.history['loss']
val_loss = history_avg_pool.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history_avg_pool.history['accuracy']
val_acc = history_avg_pool.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###################################################################

_, avg_pool_model_acc = avg_pool_model.evaluate(X_test, y_test)
print("Global avg. pool Accuracy = ", (avg_pool_model_acc * 100.0), "%")

#############################################

#Confusion matrix
#We compare labels and plot them based on correct or wrong predictions.
#Since sigmoid outputs probabilities we need to apply threshold to convert to label.

mythreshold=0.5
from sklearn.metrics import confusion_matrix
import seaborn as sns

y_pred_GAP = (avg_pool_model.predict(X_test)>= mythreshold).astype(int)
cm_GAP=confusion_matrix(y_test, y_pred_GAP)  
sns.heatmap(cm_GAP, annot=True)

########################################################
import scipy
import random

#Find a good image that shows parasite features. 
index_para = np.where(y_test == 1)[0]
img_num = random.randint(0, len(index_para-1))
img = X_test[img_num]
plt.imshow(img)

    #Get weights from the prediction layer (Only weights, not biases)
last_layer_weights = avg_pool_model.layers[-2].get_weights()[0] #Dense Prediction layer
    #Get weights for the predicted class. (In our case we defined the problem as binary and used sigmoid, so just single output)
last_layer_weights_for_pred = last_layer_weights[:, 0]

    #Get output from the last conv. layer for our input image
last_conv_model = Model(avg_pool_model.input, avg_pool_model.layers[-4].output) #Last conv layer
last_conv_output = last_conv_model.predict(img[np.newaxis,:,:,:])
last_conv_output = np.squeeze(last_conv_output)
    
    #Upsample/resize the last conv. output to same size as original image
h = (img.shape[0]/last_conv_output.shape[0])
w = (img.shape[1]/last_conv_output.shape[1])
upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)

#Multiply each feature map from the upsampled_last_conv_output with corresponding weights.
#Sum product over the last axis of both arrays, upsampled conv output and last layer weights.  
heat_map = np.dot(upsampled_last_conv_output, last_layer_weights_for_pred)
  

    #Since we have a lot of dark pixels where the edges may be thought of as 
    #high anomaly, let us drop all heat map values in this region to 0.
    #This is an optional step based on the image. 
heat_map[img[:,:,0] == 0] = 0  #All dark pixels outside the object set to 0

plt.imshow(heat_map)  


