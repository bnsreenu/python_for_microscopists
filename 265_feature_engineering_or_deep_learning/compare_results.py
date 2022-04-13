# https://youtu.be/JNjhKrwe-4k
"""
Compare results from all models. 
Here, we compare mean IOU and also IOU for class2, the difficult class 
in the sandstone data set.

We will also look at a few random images for comparison. 

"""
import tifffile as tiff
from matplotlib import pyplot as plt
from tensorflow.keras.metrics import MeanIoU
import numpy as np
import random

#Predict on a few images
#Load images and masks if not already done... 
test_image_stack = tiff.imread('data/test_images/test_sandstone_images.tif')
test_mask_stack = tiff.imread('data/test_images/test_sandstone_masks.tif')

#feature_eng_segmented_stack = tiff.imread('data/xgboost_segmented.tif')
feature_eng_segmented_stack = tiff.imread('data/lgbm_segmented.tif')
Unet_scratch_segmented_stack = tiff.imread('data/DL_segmented.tif')
VGG19_segmented_stack = tiff.imread('data/VGG19_Unet_segmented.tif')

#Using built in keras function
n_classes = 4

IOU_feature_engg = MeanIoU(num_classes=n_classes)  
IOU_feature_engg.update_state(test_mask_stack, feature_eng_segmented_stack)

IOU_Unet_from_scratch = MeanIoU(num_classes=n_classes)  
IOU_Unet_from_scratch.update_state(test_mask_stack, Unet_scratch_segmented_stack)

IOU_VGG19_pretrained = MeanIoU(num_classes=n_classes)  
IOU_VGG19_pretrained.update_state(test_mask_stack, VGG19_segmented_stack)

print("IOU - feature engineering=", IOU_feature_engg.result().numpy())
print("IOU - Unet from scratch=", IOU_Unet_from_scratch.result().numpy())
print("IOU - VGG19 pretrained=", IOU_VGG19_pretrained.result().numpy())


####################################################

def calc_class2_IOU(IOU_keras):
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    return class2_IoU

class2_IoU_feature_engg = calc_class2_IOU(IOU_feature_engg)
class2_IoU_Unet_from_scratch = calc_class2_IOU(IOU_Unet_from_scratch)
class2_IoU_VGG19_pretrained = calc_class2_IOU(IOU_VGG19_pretrained)

print("Class 2 IOU - feature engineering=", class2_IoU_feature_engg)
print("Class 2 IOU - Unet from scratch=", class2_IoU_Unet_from_scratch)
print("Class 2 IOU - VGG19 pretrained=", class2_IoU_VGG19_pretrained)

################################################################
#test_img_number = random.randint(0, test_mask_stack.shape[0]-1)
test_img_number = 15
test_img = test_image_stack[test_img_number]
ground_truth=test_mask_stack[test_img_number]

feature_eng_segmented_img = feature_eng_segmented_stack[test_img_number]
unet_scratch_segmented_img = Unet_scratch_segmented_stack[test_img_number]
VGG19_pretrained_segmented_img = VGG19_segmented_stack[test_img_number]


plt.figure(figsize=(12, 12))
plt.subplot(331)
plt.title('Testing Image')
plt.imshow(test_img[:,:], cmap='gray')
plt.subplot(332)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:], cmap='gray')
plt.subplot(333)
plt.title('Feature engineered result')
plt.imshow(feature_eng_segmented_img, cmap='gray')
plt.subplot(334)
plt.title('Unet from scratch result')
plt.imshow(unet_scratch_segmented_img, cmap='gray')
plt.subplot(335)
plt.title('VGG19 pretrained result')
plt.imshow(VGG19_pretrained_segmented_img, cmap='gray')
plt.show()