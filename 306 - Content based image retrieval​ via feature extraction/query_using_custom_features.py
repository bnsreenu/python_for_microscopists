# https://youtu.be/zN9ZINn7g24
"""
@author: DigitalSreeni

The code in this file helps you load an indexed feature database generated using
the index_using_custom_features.py file. 

Next, an image gets loaded and features extracted using the same feature extractors
used for indexing (e.g., GLCM). The features from this query image are compared against features 
features from the indexed database and a match score gets reported. The match is
performed usin gthe cosine distance method. 
https://en.wikipedia.org/wiki/Cosine_similarity

The top 3 matching image names are then printed on the screen. 

"""

import numpy as np
import h5py
from skimage import io


# read features database (h5 file)
h5f = h5py.File("CustomFeatures.h5",'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
        
#Read the query image
queryImg = io.imread("query_images/monkey4.jpg") #histo.jpg, tiger3.jpg

#Worked great on histo image as color information helps.
#But results are not good for tiger3

print(" searching for similar images")


from index_using_custom_features import extract_custom_features

#Extract Features
X = extract_custom_features(queryImg)


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
max_num_matches = 3
imlist = [imgNames[index] for i,index in enumerate(rank_ID[0:max_num_matches])]
print("top %d images in order are: " %max_num_matches, imlist)


