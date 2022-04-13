# https://youtu.be/JNjhKrwe-4k
"""

U-net based image segmentation using custom built U-net model. 
(import the model from simple_multi_unet_model file)

"""

from patchify import patchify, unpatchify
import tifffile as tiff
import numpy as np
from tensorflow.keras.utils import to_categorical
from simple_multi_unet_model import multi_unet_model #Uses softmax 
from matplotlib import pyplot as plt

img = tiff.imread('data/train_images/Sandstone_Versa0000.tif')
mask = tiff.imread('data/train_images/Sandstone_Versa0000_mask.tif')

#This will split the image into small images of shape [3,3]
img_patches = patchify(img, (128, 128), step=128)  #Step=256 for 256 patches means no overlap
mask_patches = patchify(mask, (128, 128), step=128)

all_img_patches = np.reshape(img_patches, (-1, 128,128) )
all_mask_patches = np.reshape(mask_patches, (-1, 128,128) )

print(np.unique(all_mask_patches, return_counts=True))

#Scale input images...
all_img_patches = all_img_patches / 255.
#Reshape for the neural network
all_img_patches = np.expand_dims(all_img_patches, axis=3)

#Convert masks to categorical
n_classes = 4
all_mask_patches_cat = to_categorical(all_mask_patches, num_classes=n_classes)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_img_patches, all_mask_patches_cat, test_size = 0.10, random_state = 0)

##############################################

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#If starting with pre-trained weights. 
#model.load_weights('???.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=200, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)
                    
model.save('saved_models/DL_Unet.hdf5')
#########################################################
#Evaluate the model
	# evaluate model
_, acc = model.evaluate(X_test, y_test)
print("Accuracy is = ", (acc * 100.0), "%")


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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

###########################################

#We have a very small dataset so we may not have all pixel values in test images.
#Let us check metrics on train images... 
from tensorflow.keras.models import load_model
my_model = load_model('saved_models/DL_Unet.hdf5')  

_, acc_train = my_model.evaluate(X_train, y_train)
print("Accuracy is = ", (acc_train * 100.0), "%")

_, acc_test = my_model.evaluate(X_test, y_test)
print("Accuracy is = ", (acc_test * 100.0), "%")



########################################
#Applying trained model to segment multiple files. 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
test_image_stack = test_image_stack /255.


segmented_stack=[]
for i in range(test_image_stack.shape[0]):
    print("Now segmenting image number: ", i)     #just stop here to see all file names printed
    test_img = test_image_stack[i,:,:]
    patches = patchify(test_img, (128, 128), step=128)  #Step=256 for 256 patches means no overlap

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
           # print(i,j)
            
            single_patch = patches[i,j,:,:]       
            single_patch_input = np.expand_dims(single_patch, axis=0)
            single_patch_input = np.expand_dims(single_patch_input, axis=3)
            single_patch_prediction = my_model.predict(single_patch_input)
            single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]
            output_8bit = single_patch_predicted_img.astype(np.uint8)
    
            predicted_patches.append(output_8bit)
    
    predicted_patches = np.array(predicted_patches)
    
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )
    
    reconstructed_image = unpatchify(predicted_patches_reshaped, test_img.shape)
    segmented_stack.append(reconstructed_image)

segmented_stack = np.array(segmented_stack)
tiff.imwrite('data/DL_segmented.tif', segmented_stack)

test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')
plt.imshow(test_mask_stack[0, :,:], cmap='gray')
plt.imshow(segmented_stack[0], cmap='gray')

#######################################################################

#Using built in keras function
from tensorflow.keras.metrics import MeanIoU
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
segmented_stack = tiff.imread('data/DL_segmented.tif')

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