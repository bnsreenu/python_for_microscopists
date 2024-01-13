# https://youtu.be/zN9ZINn7g24
"""
@author: DigitalSreeni

The code in this file helps you load an indexed feature database generated using
the index_using_VGG16_pretrained.py file. 

Next, an image gets loaded and features extracted using the same VGG16 network
used for indexing. The features from this query image are compared against features 
features from the indexed database and a match score gets reported. The match is
performed usin gthe cosine distance method. 
https://en.wikipedia.org/wiki/Cosine_similarity

The top 3 matching image names are then printed on the screen. 

"""
from VGG_feature_extractor import VGGNet

import numpy as np
import h5py

import matplotlib.pyplot as plt



# # read features database (h5 file)
h5f = h5py.File("VGG16Features.h5",'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
  
#Read the query image
queryImg = "query_images/monkey4.jpg"

print(" searching for similar images")

# init VGGNet16 model
model = VGGNet()

# #Extract Features
X = model.extract_feat(queryImg)


# Compute the Cosine distance between 1-D arrays
# https://en.wikipedia.org/wiki/Cosine_similarity

scores = []
from scipy import spatial
for i in range(feats.shape[0]):
    score = 1-spatial.distance.cosine(X, feats[i])
    scores.append(score)
scores = np.array(scores)   
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]


# Top 3 matches to the query image
maxres = 3
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:maxres])]
print("top %d images in order are: " %maxres, imlist)


