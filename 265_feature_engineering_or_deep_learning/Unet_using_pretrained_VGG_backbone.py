# https://youtu.be/JNjhKrwe-4k
"""
Author: Sreenivas Bhattiprolu

Multiclass semantic segmentation using U-Net

Including segmenting large images by dividing them into smaller patches 
and stiching them back
"""

import segmentation_models as sm
import numpy as np
import tensorflow

from tensorflow.keras.metrics import MeanIoU

from patchify import patchify, unpatchify
import tifffile as tiff
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

img = tiff.imread('data/train_images/Sandstone_Versa0000.tif')
mask = tiff.imread('data/train_images/Sandstone_Versa0000_mask.tif')

#This will split the image into small images of shape [3,3]
img_patches = patchify(img, (128, 128), step=128)  #Step=256 for 256 patches means no overlap
mask_patches = patchify(mask, (128, 128), step=128)

all_img_patches = np.reshape(img_patches, (-1, 128,128) )
all_mask_patches = np.reshape(mask_patches, (-1, 128,128) )

print(np.unique(all_mask_patches, return_counts=True))

#Scale input images... (Scaling will be done via preprocessing)
#all_img_patches = all_img_patches / 255.

#Reshape for the neural network
#VGG network from segmentation models expects 3 channel input image
#so, let us copy the single gray channel 3 times. 
#all_img_patches = np.expand_dims(all_img_patches, axis=3)
all_img_patches = np.stack((all_img_patches,)*3, axis=-1)

#Convert masks to categorical
n_classes = 4
all_mask_patches_cat = to_categorical(all_mask_patches, num_classes=n_classes)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_img_patches, all_mask_patches_cat, test_size = 0.10, random_state = 0)

##############################################

# IMG_HEIGHT = X_train.shape[1]
# IMG_WIDTH  = X_train.shape[2]
# IMG_CHANNELS = X_train.shape[3]
######################################################
#Reused parameters in all models

n_classes=4
activation='softmax'

LR = 0.0001
optim = tensorflow.keras.optimizers.Adam(LR)

# Segmentation models losses: Using focal loss
focal_loss = sm.losses.CategoricalFocalLoss()

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


########################################################################
#####################################################
###VGG19 model..

BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)


# define model
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation=activation)

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, focal_loss, metrics)
#model3.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)


print(model.summary())

history=model.fit(X_train, 
          y_train,
          batch_size=16, 
          epochs=200,
          verbose=1,
          validation_data=(X_test, y_test))


model.save('saved_models/vgg19_backbone_200epochs.hdf5')


##########################################################

###
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

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.show()

#####################################################

from tensorflow.keras.models import load_model

### FOR NOW LET US FOCUS ON A SINGLE MODEL

#Set compile=False as we are not loading it for training, only for prediction.
model = load_model('saved_models/vgg19_backbone_200epochs.hdf5', compile=False)


#IOU
# y_pred=model.predict(X_test)
# y_pred_argmax=np.argmax(y_pred, axis=3)

########################################
#Applying trained model to segment multiple files. 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
#test_image_stack = test_image_stack /255.


segmented_stack=[]
for i in range(test_image_stack.shape[0]):
    print("Now segmenting image number: ", i)     #just stop here to see all file names printed
    test_img = test_image_stack[i,:,:]
    patches = patchify(test_img, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            #print(i,j)
            
            single_patch = patches[i,j,:,:]       
            single_patch_input = np.expand_dims(single_patch, axis=0)
            single_patch_input = np.stack((single_patch_input,)*3, axis=-1)
            single_patch_input = preprocess_input(single_patch_input)
            single_patch_prediction = model.predict(single_patch_input)
            single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
            output_8bit = single_patch_predicted_img.astype(np.uint8)
    
            predicted_patches.append(output_8bit)
    
    predicted_patches = np.array(predicted_patches)
    
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )
    
    reconstructed_image = unpatchify(predicted_patches_reshaped, test_img.shape)
    segmented_stack.append(reconstructed_image)

segmented_stack = np.array(segmented_stack)
tiff.imwrite('data/VGG19_Unet_segmented.tif', segmented_stack)

test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')
plt.imshow(test_mask_stack[0, :,:], cmap='gray')
plt.imshow(segmented_stack[0], cmap='gray')

#######################################################################

#Using built in keras function
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_stack, segmented_stack)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)


#######################################################################
#Predict on a few images
#Load images and masks if not already done... 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')
segmented_stack = tiff.imread('data/VGG19_Unet_segmented.tif')

import random
test_img_number = random.randint(0, test_mask_stack.shape[0]-1)
#test_img_number=9
test_img = test_image_stack[test_img_number]
ground_truth=test_mask_stack[test_img_number]
segmented_img = segmented_stack[test_img_number]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(segmented_img, cmap='jet')
plt.show()