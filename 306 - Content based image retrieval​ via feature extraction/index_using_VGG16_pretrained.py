# https://youtu.be/zN9ZINn7g24
"""
@author: DigitalSreeni

This code will read images from a specified directory and extracts features
using a pre-trained VGG16 network on Imagenet database. The VGG16 network is
imported from the accompanying file VGG_feature_extractor.py where we import
the network with the top classifier layers. The final output for the feature vector
is 512. These features from every image in the input folder are captured into
a hdf5 file. This file will be imported to the query_using_VGG16_features.py 
file to search for similar images to a query image by comparing the feature 
vectors. 

"""

import os
import h5py
import numpy as np
from VGG_feature_extractor import VGGNet

images_path ="all_images/"
img_list = [os.path.join(images_path,f) for f in os.listdir(images_path)]


print("  start feature extraction ")


model = VGGNet()

path = "all_images/"

feats = []
names = []

for im in os.listdir(path):  #iterate through all images to extract features
    print("Extracting features from image - ", im)
    X = model.extract_feat(path+im)

    feats.append(X)
    names.append(im)
    
feats = np.array(feats)

# directory for storing extracted features
output = "VGG16Features.h5"

print(" writing feature extraction results to h5 file")


h5f = h5py.File(output, 'w')
h5f.create_dataset('dataset_1', data = feats)
h5f.create_dataset('dataset_2', data = np.string_(names))
h5f.close()

