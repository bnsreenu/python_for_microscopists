# https://youtu.be/hTpq9lzAb8M
"""
@author: Sreenivas Bhattiprolu

Train U-net by loading pre-trained enocoder weights. 

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
#########################################################################
#Load data for U-net training. 
#################################################################
import os
image_directory = 'data/mito/images/'
mask_directory = 'data/mito/masks/'


SIZE = 256
image_dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.  
mask_dataset = []  #Place holders to define add labels. We will add 0 to all parasitized images and 1 to uninfected.

images = os.listdir(image_directory)
for i, image_name in enumerate(images):    #Remember enumerate method adds a counter and returns the enumerate object
    if (image_name.split('.')[1] == 'tif'):
        #print(image_directory+image_name)
        image = cv2.imread(image_directory+image_name, 1)
        #image = Image.fromarray(image)
        #image = image.resize((SIZE, SIZE))
        image_dataset.append(image)

#Iterate through all images in Uninfected folder, resize to 64 x 64
#Then save into the same numpy array 'dataset' but with label 1

masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'tif'):
        image = cv2.imread(mask_directory+image_name, 0)
        #image = Image.fromarray(image)
        #image = image.resize((SIZE, SIZE))
        mask_dataset.append(image)


#Normalize images
image_dataset = np.array(image_dataset)/255.
#D not normalize masks, just rescale to 0 to 1.
mask_dataset = np.expand_dims((np.array(mask_dataset)), 3) /255.



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.20, random_state = 0)

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
########################################################################
#######################################################################
#Load unet model and load pretrained weights
from models import build_autoencoder, build_encoder, build_unet
from keras.optimizers import Adam
import segmentation_models as sm

input_shape = (256, 256, 3)
random_wt_unet_model = build_unet(input_shape)

random_wt_unet_model_weights = random_wt_unet_model.get_weights()[0][1]

pre_trained_unet_model = build_unet(input_shape)
pre_trained_unet_model.load_weights('unet_model_weights.h5')
pre_trained_unet_model_weights = pre_trained_unet_model.get_weights()[0][1]

#Load previously saved pretrained encoder weights just for comparison with the unet weights (Sanity check)
pretrained_encoder_wts = np.load('pretrained_encoder-weights.npy')

if pre_trained_unet_model_weights.all() == pretrained_encoder_wts.all():
    print("Both weights are identical")
else: 
    print("Something wrong, weghts are different")


#Compile both models, one with random weights and the other with pretrained
random_wt_unet_model.compile('Adam', loss=sm.losses.binary_focal_jaccard_loss, metrics=[sm.metrics.iou_score])
pre_trained_unet_model.compile('Adam', loss=sm.losses.binary_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

####################################################################

#train both models

#Train the model
batch_size=16

random_wt_unet_model_history = random_wt_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=25)

random_wt_unet_model.save('random_wt_unet_model_25epochs.h5')


pre_trained_unet_model_history = pre_trained_unet_model.fit(X_train, y_train, 
                    verbose=1,
                    batch_size = batch_size,
                    validation_data=(X_test, y_test ), 
                    shuffle=False,
                    epochs=25)

pre_trained_unet_model.save('pre_trained_unet_model_25epochs.h5')

########################################################################
"""
#Save history for future visualization and plotting

# convert the history.history dict to a pandas DataFrame: 
    #and save as csv
import pandas as pd    
hist_df = pd.DataFrame(random_wt_unet_model_history.history) 
hist_csv_file = 'random_wt_unet_model_history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# OR save as numpy
#Save history if you want to plot loss and IoU later
np.save('random_wt_unet_model_history.npy', random_wt_unet_model_history.history)
np.save('pre_trained_unet_model_history.npy', pre_trained_unet_model_history.history)

#Load saved history
random_wt_unet_model_history = np.load('random_wt_unet_model_history.npy', allow_pickle='TRUE').item()
pre_trained_unet_model_history = np.load('pre_trained_unet_model_history.npy', allow_pickle='TRUE').item()
"""
##########################################################################

#PLot history to see which one converged fast
history = random_wt_unet_model_history

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

acc = history.history['iou_score']
val_acc = history.history['val_iou_score']
plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()
################################################################################
#For each model check the IoU and verify few random images
from keras.models import load_model
random_wt_unet_model = load_model('random_wt_unet_model_25epochs.h5', compile=False)
                      # custom_objects={'categorical_focal_jaccard_loss': sm.losses.sm.losses.categorical_focal_jaccard_loss,
                      #                 'iou_score':sm.metrics.IOUScore(threshold=0.5)})

pre_trained_unet_model = load_model('pre_trained_unet_model_25epochs.h5', compile=False)

my_model = random_wt_unet_model
my_model = pre_trained_unet_model

import random
test_img_number = random.randint(0, X_test.shape[0]-1)
#test_img_number = 119
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]

test_img_input=np.expand_dims(test_img, 0)
prediction = (my_model.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)

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


#Calculate IoU for all test images and average
 
import pandas as pd

IoU_values = []
for img in range(0, X_test.shape[0]):
    temp_img = X_test[img]
    ground_truth=y_test[img]
    temp_img_input=np.expand_dims(temp_img, 0)
    prediction = (my_model.predict(temp_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
    
    IoU = MeanIoU(num_classes=n_classes)
    IoU.update_state(ground_truth[:,:,0], prediction)
    IoU = IoU.result().numpy()
    IoU_values.append(IoU)

    print(IoU)
    


df = pd.DataFrame(IoU_values, columns=["IoU"])
df = df[df.IoU != 1.0]    
mean_IoU = df.mean().values
print("Mean IoU is: ", mean_IoU)    
    
