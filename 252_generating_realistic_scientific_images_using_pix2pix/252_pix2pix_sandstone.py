# https://youtu.be/my7LEgYTJto

"""
Generating realistic looking scientific images using Pix2Pix GAN

Dataset link: https://drive.google.com/file/d/1HWtBaSa-LTyAMgf2uaz1T9o1sTWDBajU/view
(Please read the Readme document in the dataset folder for more information. )

"""


from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot 


#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
#n_classes=4 #Number of classes for segmentation

#Capture training image info as a list
tar_images = []

for directory_path in glob.glob("sandstone/images/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.tif")):
        img = cv2.imread(img_path, 1)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        tar_images.append(img)
       
#Convert list to array for machine learning processing        
tar_images = np.array(tar_images)

#Capture mask/label info as a list
src_images = [] 
for directory_path in glob.glob("sandstone/masks/"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.tif")):
        mask = cv2.imread(mask_path, 1)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        src_images.append(mask)
        
#Convert list to array for machine learning processing          
src_images = np.array(src_images)

print(np.unique(src_images))



n_samples = 3
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + i)
	pyplot.axis('off')
	pyplot.imshow(src_images[i])
# plot target image
for i in range(n_samples):
	pyplot.subplot(2, n_samples, 1 + n_samples + i)
	pyplot.axis('off')
	pyplot.imshow(tar_images[i])
pyplot.show()

#######################################################

from pix2pix_model import define_discriminator, define_generator, define_gan, train
# define input shape based on the loaded dataset
image_shape = src_images.shape[1:]
# define the models
d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
# define the composite model
gan_model = define_gan(g_model, d_model, image_shape)

#Define data
# load and prepare training images
data = [src_images, tar_images]

#Preprocess data to change input range to values between -1 and 1
# This is because the generator uses tanh activation in the output layer
#And tanh ranges between -1 and 1
def preprocess_data(data):
	# load compressed arrays
	# unpack arrays
	X1, X2 = data[0], data[1]
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

dataset = preprocess_data(data)

from datetime import datetime 
start1 = datetime.now() 

train(d_model, g_model, gan_model, dataset, n_epochs=50, n_batch=1) 
#Reports parameters for each batch (total 1600) for each epoch.
#For 10 epochs we should see 16000

stop1 = datetime.now()
#Execution time of the model 
execution_time = stop1-start1
print("Execution time is: ", execution_time)

#Reports parameters for each batch (total 1096) for each epoch.
#For 10 epochs we should see 10960
g_model.save('sandstone_.h5')
#########################################
#Test trained model on a few images...

from keras.models import load_model
from numpy.random import randint
from numpy import vstack

model = load_model('sandstone_50epochs.h5')

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Input-segm-img', 'Output-Generated', 'Original']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		pyplot.subplot(1, 3, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(images[i,:,:,0], cmap='gray')
		# show title
		pyplot.title(titles[i])
	pyplot.show()



[X1, X2] = dataset
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)

###########################################
test_src_img = cv2.imread("sandstone/test_mask.tif", 1)       
test_src_img = cv2.resize(test_src_img, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
test_src_img = (test_src_img - 127.5) / 127.5
test_src_img = np.expand_dims(test_src_img, axis=0)

# generate image from source
gen_test_image = model.predict(test_src_img)

#pyplot.imshow(test_src_img[0, :,:,0], cmap='gray')
pyplot.imshow(gen_test_image[0, :,:,0], cmap='gray')
