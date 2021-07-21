# https://youtu.be/ZoJuhRbzEiM
"""
Mitochondria semantic segmentation using U-net, Attention Unet and R2 Unet
and others using keras-unet-collection library.
# https://github.com/yingkaisha/keras-unet-collection

Author: Dr. Sreenivas Bhattiprolu

Dataset from: https://www.epfl.ch/labs/cvlab/data/data-em/
Images and masks are divided into patches of 256x256. 
"""

import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam
#import tensorflow as tf
from datetime import datetime 
import cv2
from PIL import Image
#from keras import backend, optimizers


# force channels-first ordering for all loaded images
#backend.set_image_data_format('channels_last')  #The models are designed to use channels first


image_directory = 'data/images/'
mask_directory = 'data/masks/'


SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 1)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        image_dataset.append(np.array(image))

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))


#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#Sanity check, view few mages
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (256, 256, 3)), cmap='gray')
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (256, 256)), cmap='gray')
plt.show()

#######################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]
num_labels = 1  #Binary
input_shape = (IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)
batch_size = 8
#FOCAL LOSS AND DICE METRIC
#Focal loss helps focus more on tough to segment classes.
#from focal_loss import BinaryFocalLoss

###############################################################################
#Try various models: Unet, Attention_UNet, and Attention_ResUnet


from keras_unet_collection import models, losses
###############################################################################
#Model 1: Unet with ImageNet trained VGG16 backbone
help(models.unet_2d)

model_Unet = models.unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='unet')


model_Unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_Unet.summary())

start1 = datetime.now() 

Unet_history = model_Unet.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop1 = datetime.now()
#Execution time of the model 
execution_time_Unet = stop1-start1
print("UNet execution time is: ", execution_time_Unet)

model_Unet.save('mitochondria_unet_collection_UNet_50epochs.hdf5')
#############################################################
# Unet Plus
help(models.unet_plus_2d)

model_Unet_plus = models.unet_plus_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='unet_plus')


model_Unet_plus.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_Unet_plus.summary())

start2 = datetime.now() 

Unet_plus_history = model_Unet_plus.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop2 = datetime.now()
#Execution time of the model 
execution_time_Unet_plus = stop2-start2
print("UNet plus execution time is: ", execution_time_Unet_plus)

model_Unet_plus.save('mitochondria_unet_collection_UNet_plus_50epochs.hdf5')
##############################################################################
#Attention U-net with an ImageNet-trained backbone

help(models.att_unet_2d)

model_att_unet = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           atten_activation='ReLU', attention='add', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=False, unpool=False, 
                           backbone='VGG16', weights='imagenet', 
                           freeze_backbone=True, freeze_batch_norm=True, 
                           name='attunet')


model_att_unet.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_att_unet.summary())

start3 = datetime.now() 

att_unet_history = model_att_unet.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop3 = datetime.now()
#Execution time of the model 
execution_time_att_Unet = stop3-start3
print("Attention UNet execution time is: ", execution_time_att_Unet)

model_att_unet.save('mitochondria_unet_collection_att_UNet_50epochs.hdf5')

#######################################################################
#Without loading weights
#####################################################################
#Model 4: Unet with ImageNet trained VGG16 backbone
help(models.unet_2d)

model_Unet_from_scratch = models.unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           backbone=None, weights=None, 
                           freeze_backbone=False, freeze_batch_norm=False, 
                           name='unet')


model_Unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_Unet_from_scratch.summary())

start4 = datetime.now() 

Unet_from_scratch_history = model_Unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop4 = datetime.now()
#Execution time of the model 
execution_time_Unet_from_scratch = stop4-start4
print("UNet from scratch execution time is: ", execution_time_Unet_from_scratch)

model_Unet_from_scratch.save('mitochondria_unet_collection_UNet_from_scratch_50epochs.hdf5')

####################################################################################
#Model 5: Recurrent Residual (R2) U-Net
help(models.r2_unet_2d)

model_r2_Unet_from_scratch = models.r2_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           recur_num=2,
                           activation='ReLU', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           name='r2_unet')


model_r2_Unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_r2_Unet_from_scratch.summary())

start5 = datetime.now() 

r2_Unet_from_scratch_history = model_r2_Unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop5 = datetime.now()
#Execution time of the model 
execution_time_r2_Unet_from_scratch = stop5-start5
print("R2 UNet from scratch execution time is: ", execution_time_r2_Unet_from_scratch)

model_r2_Unet_from_scratch.save('mitochondria_unet_collection_r2_UNet_from_scratch_50epochs.hdf5')

############################################################################
#Model 6: Attention Unet from scratch - no backbone or weights.
help(models.att_unet_2d)

model_att_unet_from_scratch = models.att_unet_2d((256, 256, 3), filter_num=[64, 128, 256, 512, 1024], 
                           n_labels=num_labels, 
                           stack_num_down=2, stack_num_up=2, 
                           activation='ReLU', 
                           atten_activation='ReLU', attention='add', 
                           output_activation='Sigmoid', 
                           batch_norm=True, pool=True, unpool=True, 
                           backbone=None, weights=None, 
                           freeze_backbone=False, freeze_batch_norm=False, 
                           name='attunet')


model_att_unet_from_scratch.compile(loss='binary_crossentropy', optimizer=Adam(lr = 1e-3), 
              metrics=['accuracy', losses.dice_coef])

print(model_att_unet_from_scratch.summary())

start6 = datetime.now() 

att_unet_from_scratch_history = model_att_unet_from_scratch.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=50)

stop6 = datetime.now()
#Execution time of the model 
execution_time_att_Unet_from_scratch = stop6-start6
print("Attention UNet from scratch execution time is: ", execution_time_att_Unet_from_scratch)

model_att_unet_from_scratch.save('mitochondria_unet_collection_att_UNet_from_scratch_50epochs.hdf5')
############################################################################
# convert the history.history dict to a pandas DataFrame and save as csv for
# future plotting
import pandas as pd    
unet_history_df = pd.DataFrame(Unet_history.history) 
unet_plus_history_df = pd.DataFrame(Unet_plus_history.history) 
att_unet_history_df = pd.DataFrame(att_unet_history.history) 

unet_from_scratch_history_df = pd.DataFrame(Unet_from_scratch_history.history) 
r2_Unet_from_scratch_history_df = pd.DataFrame(r2_Unet_from_scratch_history.history) 
att_unet_from_scratch_history_df = pd.DataFrame(att_unet_from_scratch_history.history) 

with open('unet_history_df.csv', mode='w') as f:
    unet_history_df.to_csv(f)
    
with open('unet_plus_history_df.csv', mode='w') as f:
    unet_plus_history_df.to_csv(f)

with open('att_unet_history_df.csv', mode='w') as f:
    att_unet_history_df.to_csv(f)    

with open('unet_from_scratch_history_df.csv', mode='w') as f:
    unet_from_scratch_history_df.to_csv(f)    
    
with open('r2_Unet_from_scratch_history_df.csv', mode='w') as f:
    r2_Unet_from_scratch_history_df.to_csv(f)    

with open('att_unet_from_scratch_history_df.csv', mode='w') as f:
    att_unet_from_scratch_history_df.to_csv(f)        


#######################################################################
#Check history plots, one model at a time
history = Unet_history
history = Unet_plus_history
history = att_unet_history
history = Unet_from_scratch_history
history = r2_Unet_from_scratch_history
history = att_unet_from_scratch_history

#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['dice_coef']
#acc = history.history['accuracy']
val_acc = history.history['val_dice_coef']
#val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Dice')
plt.plot(epochs, val_acc, 'r', label='Validation Dice')
plt.title('Training and validation Dice')
plt.xlabel('Epochs')
plt.ylabel('Dice')
plt.legend()
plt.show()

#######################################################



model = model_Unet
model = model_Unet_plus
model = model_att_unet

model = model_Unet_from_scratch
model = model_r2_Unet_from_scratch
model = model_att_unet_from_scratch

#Load one model at a time for testing.
#model = tf.keras.models.load_model(model, compile=False)


import random
test_img_number = random.randint(0, X_test.shape[0]-1)  #Test with 119

test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')

plt.show()


#IoU for a single image
from tensorflow.keras.metrics import MeanIoU
n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(ground_truth[:,:,0], prediction)
print("Mean IoU =", IOU_keras.result().numpy())


#Calculate IoU and average
 
import pandas as pd

IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    #print(IoU)
    


df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)    
    



