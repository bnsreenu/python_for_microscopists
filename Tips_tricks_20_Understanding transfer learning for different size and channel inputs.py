# https://youtu.be/5kbpoIQUB4Q
"""
The input to the Vgg 16 model is 224x224x3 pixels images. 
The Kernel size is 3x3 and the pool size is 2x2 for all the layers.

If our image size is different, can we still use transfer learning?
The answer is YES.

Input image size does not matter as the weights are associated with the filter 
kernel size. This does not change based on the input image size, for convolutional layers. 
The number of channels does matter, as it affects the number of weights for the first convolutional layer. 
We can still use transfer learning by copying weights for the first channels from the original model
and then filling the additional channel weights with the mean of existing weights along the channels. 

"""


from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
from tensorflow.keras.models import Model

######################################
#Verify how trainable parameters for each layer changes by changing the shape
#and also number of channels. 
#Start with 224x224x3 and compare with the same when imported from keras (with top=false)
#Both should be identical. 
#You will notice that changing h and w does not have an affect as it does not change the filter size.
#Changing the number of channels will change the number of trainable weights, as it affects the input layer.  
#Therefore, transfer learning will only work for different size images and not channels. 

#VGG16 model

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten

model = Sequential()

model.add(Conv2D(input_shape=(256,256,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=10, activation="softmax")) 

print(model.summary())


################################

#Let us import VGG16 model with imagenet weights from keras.

img1_shape = (224,224,3)
model_224 = VGG16(include_top=False, weights='imagenet', input_shape=img1_shape)
print(model_224.summary())
#plot_model(model_224, to_file='model_224.png', show_shapes=True, show_layer_names=True)


img2_shape = (256,256,3)
model_256 = VGG16(include_top=False, weights='imagenet', input_shape=img2_shape)
print(model_256.summary())


img3_shape = (1024,1024,3)
model_1024 = VGG16(include_top=False, weights='imagenet', input_shape=img3_shape)
print(model_1024.summary())

#Different channels, will throw an error
img1b_shape = (224,224,1)
model_224b = VGG16(include_top=False, weights='imagenet', input_shape=img1b_shape)
print(model_224b.summary())

###########################################
#What if you want to use pretrained weights (Xfer learning) for images
#with different number of channels or single channel?
#Let us say our new image has only 1 channel - Grayscale..
############################################################################
###################################################################

#Import vgg model by not defining an input shape. 
vgg_model = VGG16(include_top=False, weights='imagenet')
print(vgg_model.summary())

#Get the dictionary of config for vgg16
vgg_config = vgg_model.get_config()

# Change the input shape to new desired shape
h, w, c = 1024, 1024, 1
vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)


#Create new model with the updated configuration
vgg_updated = Model.from_config(vgg_config)
print(vgg_updated.summary())

# Check Weights of first conv layer in the original model...
orig_model_conv1_block1_wts = vgg_model.layers[1].get_weights()[0]

print(orig_model_conv1_block1_wts[:,:,0,0])
print(orig_model_conv1_block1_wts[:,:,1,0])
print(orig_model_conv1_block1_wts[:,:,2,0])

# Check Weights of first conv layer in the new model...
new_model_conv1_block1_wts = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts[:,:,0,0])
#Notice the Random weights....

#Let us average weights for all RGB channels for the first convolutional layer
#and assign it to the first conv layer in our new model. 

# Function that calculates average of weights along the channel axis  
def avg_wts(weights):  
  average_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)  #Find mean along the channel axis (second to last axis)
  return(average_weights)

#Get the configuration for the updated model and extract layer names. 
#We will use these names to copy over weights from the original model. 
vgg_updated_config = vgg_updated.get_config()
vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in range(len(vgg_updated_config['layers']))]

#Name of the first convolutional layer.
#Remember that this is the only layer with new additional weights. All other layers
#will have same weights as the original model. 
first_conv_name = vgg_updated_layer_names[1]

#Update weights for all layers. And for the first conv layer replace weights with average of all 3 channels. 
for layer in vgg_model.layers:
    if layer.name in vgg_updated_layer_names:
     
      if layer.get_weights() != []:  #All convolutional layers and layers with weights (no input layer or any pool layers)
        target_layer = vgg_updated.get_layer(layer.name)
    
        if layer.name in first_conv_name:    #For the first convolutionl layer
          weights = layer.get_weights()[0]
          biases  = layer.get_weights()[1]
          
          weights_single_channel = avg_wts(weights)
                                                    
          target_layer.set_weights([weights_single_channel, biases])  #Now set weights for the first conv. layer
          target_layer.trainable = False   #You can make this trainable if you want. 
    
        else:
          target_layer.set_weights(layer.get_weights())   #Set weights to all other layers. 
          target_layer.trainable = False  #You can make this trainable if you want. 


# Check Weights of first conv layer in the new model...
#Compare against the original model weights
new_model_conv1_block1_wts_updated = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts_updated[:,:,0,0])


######################################################################

#Now, let us see how we can do the same but for an input image with many channels. 
#Let us say our new image has 9 channels. 
#########################################################################
#Notice that the additional weights come from the first conv. layer only. 
# Let us start by copying the config information of the VGG model. 
#This way we can easily edit the model input by copying it into a new model.

#Import vgg model by not defining an input shape. 
vgg_model = VGG16(include_top=False, weights='imagenet')
print(vgg_model.summary())

#Get the dictionary of config for vgg16
vgg_config = vgg_model.get_config()

# Change the input shape to new desired shape
h, w, c = 1024, 1024, 9
vgg_config["layers"][0]["config"]["batch_input_shape"] = (None, h, w, c)


#Create new model with the updated configuration
vgg_updated = Model.from_config(vgg_config)
print(vgg_updated.summary())

# Check Weights of first conv layer in the original model...
orig_model_conv1_block1_wts = vgg_model.layers[1].get_weights()[0]

print(orig_model_conv1_block1_wts[:,:,0,0])
print(orig_model_conv1_block1_wts[:,:,1,0])
print(orig_model_conv1_block1_wts[:,:,2,0])

# Check Weights of first conv layer in the new model...
new_model_conv1_block1_wts = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts[:,:,0,0])
print(new_model_conv1_block1_wts[:,:,1,0])
print(new_model_conv1_block1_wts[:,:,2,0])
print(new_model_conv1_block1_wts[:,:,3,0])
print(new_model_conv1_block1_wts[:,:,4,0])
#Random weights....


#New model created with updated input shape but weights are not copied from the original input.

#Since we have more channels to our input layer, we need to either randomly
#assign weights or get an average of all existing weights from the input layer
#and assign to the new channels as starting point. 
#Assigning average of weights may be a better approach.

# Function that calculates average of weights along the channel axis and then
#copies it over n number of times. n being the new channels that need to be concatenated with the original channels. 
def avg_and_copy_wts(weights, num_channels_to_fill):  #num_channels_to_fill are the extra channels for which we need to fill weights
  average_weights = np.mean(weights, axis=-2).reshape(weights[:,:,-1:,:].shape)  #Find mean along the channel axis (second to last axis)
  wts_copied_to_mult_channels = np.tile(average_weights, (num_channels_to_fill, 1)) #Repeat (copy) the array multiple times
  return(wts_copied_to_mult_channels)

#Get the configuration for the updated model and extract layer names. 
#We will use these names to copy over weights from the original model. 
vgg_updated_config = vgg_updated.get_config()
vgg_updated_layer_names = [vgg_updated_config['layers'][x]['name'] for x in range(len(vgg_updated_config['layers']))]

#Name of the first convolutional layer.
#Remember that this is the only layer with new additional weights. All other layers
#will have same weights as the original model. 
first_conv_name = vgg_updated_layer_names[1]

#Update weights for all layers. And for the first conv layer, copy the first
#three layer weights and fill others with the average of all three. 
for layer in vgg_model.layers:
    if layer.name in vgg_updated_layer_names:
     
      if layer.get_weights() != []:  #All convolutional layers and layers with weights (no input layer or any pool layers)
        target_layer = vgg_updated.get_layer(layer.name)
    
        if layer.name in first_conv_name:    #For the first convolutionl layer
          weights = layer.get_weights()[0]
          biases  = layer.get_weights()[1]
    
          weights_extra_channels = np.concatenate((weights,   #Keep the first 3 channel weights as-is and copy the weights for additional channels.
                                                  avg_and_copy_wts(weights, c - 3)),  # - 3 as we already have weights for the 3 existing channels in our model. 
                                                  axis=-2)
                                                  
          target_layer.set_weights([weights_extra_channels, biases])  #Now set weights for the first conv. layer
          target_layer.trainable = False   #You can make this trainable if you want. 
    
        else:
          target_layer.set_weights(layer.get_weights())   #Set weights to all other layers. 
          target_layer.trainable = False  #You can make this trainable if you want. 


# Check Weights of first conv layer in the new model...
#Compare against the original model weights
new_model_conv1_block1_wts_updated = vgg_updated.layers[1].get_weights()[0]
print(new_model_conv1_block1_wts_updated[:,:,0,0])
print(new_model_conv1_block1_wts_updated[:,:,1,0])
print(new_model_conv1_block1_wts_updated[:,:,2,0])
print(new_model_conv1_block1_wts_updated[:,:,3,0])
print(new_model_conv1_block1_wts_updated[:,:,4,0])
