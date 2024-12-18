# https://youtu.be/dCcRWdmmgA0

"""
Disclaimer: I took ChatGPT help for PCA and 3D plotting. Was feeling lazy to even
look at my old code and copy. ChatGPT gave me usable code on the 3rd try, not bad!!!  

"""


from VGG_feature_extractor import VGGNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def plot_feature_space(features, query_feat, top_matches, imgNames, title="Feature Space Visualization"):
    """Visualize high-dimensional features in 3D using PCA"""
    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    # Combine query and database features
    all_features = np.vstack([features, query_feat.reshape(1, -1)])
    features_3d = pca.fit_transform(all_features)
    
    # Split back into database and query
    db_features_3d = features_3d[:-1]
    query_feature_3d = features_3d[-1]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all database points
    ax.scatter(db_features_3d[:, 0], db_features_3d[:, 1], db_features_3d[:, 2], 
              c='blue', alpha=0.5, label='Database features')
    
    # Plot query point
    ax.scatter(query_feature_3d[0], query_feature_3d[1], query_feature_3d[2], 
              c='red', s=100, label='Query image')
    
    # Plot top matches
    matches_3d = db_features_3d[top_matches]
    ax.scatter(matches_3d[:, 0], matches_3d[:, 1], matches_3d[:, 2], 
              c='green', s=100, label='Top matches')
    
    # Add labels for top matches
    for i, match_idx in enumerate(top_matches):
        img_name = imgNames[match_idx].decode('utf-8') if isinstance(imgNames[match_idx], bytes) else imgNames[match_idx]
        ax.text(matches_3d[i, 0], matches_3d[i, 1], matches_3d[i, 2], f'{i+1}. {img_name}')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    ax.legend()
    plt.show()

# Your existing code...
h5f = h5py.File("CNNFeatures.h5",'r')
feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()

queryImg = "query_images/histo.jpg"
model = VGGNet()
query_feat = model.extract_feat(queryImg)

scores = []
from scipy import spatial
for i in range(feats.shape[0]):
    score = 1-spatial.distance.cosine(query_feat, feats[i])
    scores.append(score)
scores = np.array(scores)   
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

# Get top 3 matches
top_n = 3
top_matches = rank_ID[:top_n]
top_scores = rank_score[:top_n]

# Print matches
print(f"Top {top_n} matches with similarity scores:")
for i, (image_id, score) in enumerate(zip(top_matches, top_scores)):
    image_name = imgNames[image_id].decode('utf-8') if isinstance(imgNames[image_id], bytes) else imgNames[image_id]
    print(f"{i+1}. Image: {image_name}, Score: {score:.4f}")

# Visualize feature space
plot_feature_space(feats, query_feat, top_matches, imgNames, 
                  "VGG16 Feature Space - Image Retrieval Results")