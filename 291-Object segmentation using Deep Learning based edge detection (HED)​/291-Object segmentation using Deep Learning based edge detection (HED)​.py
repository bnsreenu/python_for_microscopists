# https://youtu.be/un7QvhXZ_G4
"""

Original HED papr: https://arxiv.org/pdf/1504.06375.pdf

Caffe model is encoded into two files
1. Proto text file: https://github.com/s9xie/hed/blob/master/examples/hed/deploy.prototxt
2. Pretrained caffe model: http://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel
NOTE: In future, if these links do not work, I cannot help. Please Google 
and find updated links (information current as of October 2022)

Steps for edge detection followed by connected components-based labeling
for object segmentation:
    
1. Define the crop layer (not implemented by default) ​
2. Define the network and load the pre-trained model.​
3. Register the crop layer with the network
4. Create blob from the image – basically create a preprocessed image​
5. Load pretrained model (you need both the proto text and caffe model files)​
6. Pass the blob image through model​ (forward pass)
7. Get output​
8. Get image ready for connected components (blur, threshold)​
9. Perform connected components based labeling​
10. (Optional) draw markers for visualization purposes​
11. (Optional) filter out small objects​
12. Export your data​


"""
import cv2
from matplotlib import pyplot as plt
import numpy as np


# There is a Crop layer that the HED network uses which is not implemented by 
# default so we need to provide our own implementation of this layer.
#Without the crop layer, the final result will be shifted to the right and bottom
#cropping part of the image
class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]


# The pre-trained model that OpenCV uses has been trained in Caffe framework
#Download from the link above
protoPath = "hed_model/deploy.prototxt"
modelPath = "hed_model/hed_pretrained_bsds.caffemodel"
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# register our crop layer with the model
cv2.dnn_registerLayer("Crop", CropLayer)


# load the input image and grab its dimensions, for future use while defining the blob
img = cv2.imread("pebbles.jpg")
plt.imshow(img)
(H, W) = img.shape[:2]

# construct a blob out of the input image 
#blob is basically preprocessed image. 
#OpenCV’s new deep neural network (dnn ) module contains two functions that 
#can be used for preprocessing images and preparing them for 
#classification via pre-trained deep learning models.
# It includes scaling and mean subtraction
#How to calculate the mean?
mean_pixel_values= np.average(img, axis = (0,1))
blob = cv2.dnn.blobFromImage(img, scalefactor=0.7, size=(W, H),
                             #mean=(mean_pixel_values[0], mean_pixel_values[1], mean_pixel_values[2]),
                             mean=(105, 117, 123),
                             swapRB= False, crop=False)

#View image after preprocessing (blob)
blob_for_plot = np.moveaxis(blob[0,:,:,:], 0,2)
plt.imshow(blob_for_plot)


# set the blob as the input to the network and perform a forward pass
# to compute the edges
net.setInput(blob)
hed = net.forward()
hed = hed[0,0,:,:]  #Drop the other axes 
#hed = cv2.resize(hed[0, 0], (W, H))
hed = (255 * hed).astype("uint8")  #rescale to 0-255

plt.imshow(hed, cmap='gray')

####################
#Connected component based labeling

# Load segmented binary image, Gaussian blur, grayscale, Otsu's threshold
blur = cv2.GaussianBlur(hed, (3,3), 0)

thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.imshow(thresh)

# Perform connected component labeling
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=4)

# Create false color image with black background and colored objects
colors = np.random.randint(0, 255, size=(n_labels, 3), dtype=np.uint8)
colors[0] = [0, 0, 0]  # black background
false_colors = colors[labels]
plt.imshow(false_colors)

# Obtain centroids
false_colors_centroid = false_colors.copy()
for centroid in centroids:
    cv2.drawMarker(false_colors_centroid, (int(centroid[0]), int(centroid[1])),
                   color=(255, 255, 255), markerType=cv2.MARKER_CROSS)
plt.imshow(false_colors_centroid)

# Remove small objects
MIN_AREA = 50
false_colors_area_filtered = false_colors.copy()
for i, centroid in enumerate(centroids[1:], start=1):
    area = stats[i, 4]
    if area > MIN_AREA:
        cv2.drawMarker(false_colors_area_filtered, (int(centroid[0]), int(centroid[1])),
                       color=(255, 255, 255), markerType=cv2.MARKER_CROSS)

plt.figure(figsize=(12, 12))
plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(hed)
plt.subplot(223)
plt.imshow(thresh)
plt.subplot(224)
plt.imshow(false_colors_area_filtered) 
plt.show()

############################
#Alternatively, We can also use regionprops from skimage to extract various parameters

# regionprops function in skimage measure module calculates useful parameters for each object.
from skimage import measure
props = measure.regionprops_table(labels, intensity_image=img, 
                              properties=['label',
                                          'area', 'equivalent_diameter',
                                          'mean_intensity', 'solidity'])
    
import pandas as pd
df = pd.DataFrame(props)

#Filter by size
df = df[df.area > 50]
df = df[df.area < 10000]

print(df.head())
