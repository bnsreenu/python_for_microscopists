# https://youtu.be/7IL7LKSLb9I
"""
pip install patchify
"""

import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify
import tifffile as tiff

large_image_stack = tiff.imread('small_dataset_for_training/images/12_training_mito_images.tif')
large_mask_stack = tiff.imread('small_dataset_for_training/masks/12_training_mito_masks.tif')

for img in range(large_image_stack.shape[0]):

    large_image = large_image_stack[img]
    
    patches_img = patchify(large_image, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:]
            tiff.imwrite('patches/images/' + 'image_' + str(img) + '_' + str(i)+str(j)+ ".tif", single_patch_img)
            

for msk in range(large_mask_stack.shape[0]):
     
    large_mask = large_mask_stack[msk]
    
    patches_mask = patchify(large_mask, (256, 256), step=256)  #Step=256 for 256 patches means no overlap
    

    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            
            single_patch_mask = patches_mask[i,j,:,:]
            tiff.imwrite('patches/masks/' + 'mask_' + str(msk) + '_' + str(i)+str(j)+ ".tif", single_patch_mask)
            single_patch_mask = single_patch_mask / 255.
            

