# https://youtu.be/iuQ_f3W5Ttk
"""
Dataset from: Dataset from: https://susanqq.github.io/UTKFace/ 

Latent space is hard to interpret unless conditioned using many classes.​
But, the latent space can be exploited using generated images.​
Here is how...

x Generate 10s of images using random latent vectors.​
x Identify many images within each category of interest (e.g., smiling man, neutral man, etc. )​
x Average the latent vectors for each category to get a mean representation in the latent space (for that category).​
x Use these mean latent vectors to generate images with features of interest. ​

This part of the code is used to generate 128x128x3 images (of faces) using a trained
generator model. 
Alo, faces can be generated using two random latent vectors and interpolated in between.
Finally, the features in the new images can be 'engineered' by doing simple arithmetic
between vectors that are used to generate images. 
In summary, you can find the latent vectors for Smiling Man, neutral face man, 
and a baby with neutral face and then generate a smiling baby face by:
    Smiling Man + Neutral Man - Neutral baby = Smiling Baby
    
"""

from numpy import asarray
from numpy.random import randn
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt


#####################################################################
#Let us start by generating images using random latent vectors.
#########################################################################
# Function to generate random latent points
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim) #Reshape to be provided as input to the generator.
	return z_input

# Function to create a plot of generated images
def plot_generated(examples, n):
	# plot images
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :])
	plt.show()

# load the saved model
model = load_model('saved_data_during_training/models/generator_model_128x128_100.h5')
# generate latent vectors to be used as input to the generator
#Here, we are generating 25 latent vectors
latent_points = generate_latent_points(100, 25)
# generate images using the loaded generator model
X  = model.predict(latent_points)
# scale from [-1,1] to [0,1] for plotting
X = (X + 1) / 2.0
# plot the generated images. Let us do 5x5 plot as we generated 25 images
plot_generated(X, 5)

#####################################################################
#Now, let us generate 2 latent vectors and interpolate between them.
#Let us do linear interpolation although in reality the latent space is curved. 
#Interpolating between faces - Linear interpolation
#################################################################

from numpy import linspace

# Function to generate random latent points
#Same as defined above, re-defining for convenience. 
def generate_latent_points(latent_dim, n_samples, n_classes=10):
 	# generate points in the latent space
 	x_input = randn(latent_dim * n_samples)
 	z_input = x_input.reshape(n_samples, latent_dim) #Reshape to be provided as input to the generator.
 	return z_input

# Interpolation between two points in latent space
def interpolate_points(p1, p2, n_steps=10):
 	# interpolate between points (e.g., between 0 and 1 if you divde to 10 then you have 0.111, 0.222, etc.)
 	ratios = linspace(0, 1, num=n_steps)
 	# linear interpolation of vectors based on the above interpolation ratios
 	vectors = list()
 	for ratio in ratios:
         v = (1.0 - ratio) * p1 + ratio * p2
         vectors.append(v)
 	return asarray(vectors)

# create a plot of generated images
def plot_generated(examples, n):
 	# plot images
 	for i in range(n):
         plt.subplot(1, n, 1 + i)
         plt.axis('off')
         plt.imshow(examples[i, :, :])
 	plt.show()

# load the model, if you haven't already loaded it above. 
model = load_model('saved_data_during_training/models/generator_model_128x128_100.h5')
# generate points in latent space
#Let us generate 2 latent points between which we will interpolate
pts = generate_latent_points(100, 2)
# interpolate points in latent space
interpolated = interpolate_points(pts[0], pts[1])
# generate images using the interpolated latent points
X = model.predict(interpolated)
# scale from [-1,1] to [0,1] for plotting
X = (X + 1) / 2.0
# plot the result
plot_generated(X, len(interpolated))

################################################################
#Now, let us perform arithmetic with latent points so we can generate faces
#with features of interest. 
#To work with latent points we must first generate a bunch of faces and 
#save them along with their corresponding latent points. This can be used
#to visually locate images of interest and thus identify the latent points.
#For example, latent points corresponding to baby face or sun glasses, etc. 
###########################################################

from numpy import mean, expand_dims
# example of loading the generator model and generating images

# Function to generate random latent points
#Same as defined above, re-defining for convenience. 
def generate_latent_points(latent_dim, n_samples, n_classes=10):
	x_input = randn(latent_dim * n_samples)
	z_input = x_input.reshape(n_samples, latent_dim) #Reshape to be provided as input to the generator.
	return z_input

# create a plot of generated images and save for easy visualization
def plot_generated(examples, n):
    plt.figure(figsize=(16, 16))
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :])
    plt.savefig('generated_faces.png')
    plt.close()

# load the model, if you haven't already loaded it above
model = load_model('saved_data_during_training/models/generator_model_128x128_100.h5')

# generate points in latent space that we can use to generate some images
#We then identify some images with our features of interest and locate thir corresponding latent vectors
latent_points = generate_latent_points(100, 100)

#Plot the latent points to see that they are spread around and we have no clue how to interpret them. 
import seaborn as sns
sns.scatterplot(latent_points[0], latent_points[1])

# generate images using the latent points. 
X  = model.predict(latent_points)
# scale from [-1,1] to [0,1] for plotting
X = (X + 1) / 2.0
# plot and save generated images
plot_generated(X, 10)

#Now, identify images corresponding to a specific type.
#e.g. all baby face images, smiling man images, 
# smiling man - neutral man + baby face = smiling baby

# retrieve specific points
#Now, identify images corresponding to a specific type.
#Start counting from 1 as we are going to offset our image number later, by subtracting 1.
#e.g. all baby face images, smiling man images, 
# smiling man - neutral man + baby face = smiling baby
#OR try adult with glasses  - adult no glasses + baby no glasses

#Identify a few images from classes of interest
# smiling_man_ix = [1, 10, 16, 26, 27, 28]
# neutral_man_ix = [16, 95, 63]
# baby_ix = [13,26,28,93,94]
adult_with_glasses = [3,39,40]
adult_no_glasses = [4, 7, 8]
#baby_no_glasses = [15,20]
person_with_lipstick = [9, 10, 11, 31]
#person_no_lipstick = [1, 4, 9, 15]

#Reassign classes of interest to new variables... just to make it easy not
# to change names all the time we get interested in new features. 
feature1_ix = adult_with_glasses
feature2_ix = adult_no_glasses
feature3_ix = person_with_lipstick

# Function to average list of latent space vectors to get the mean for a given type
def average_points(points, ix):
	# subtract 1 from image index so it matches the image from the array
    # we are doing this as our array starts at 0 but we started counting at 1. 
	zero_ix = [i-1 for i in ix]
	# retrieve required vectors corresponding to the selected images
	vectors = points[zero_ix]
	# average the vectors
	avg_vector = mean(vectors, axis=0)
	
	return avg_vector

# average vectors for each class
feature1 = average_points(latent_points, feature1_ix)
feature2 = average_points(latent_points, feature2_ix)
feature3 = average_points(latent_points, feature3_ix)

# Vector arithmetic....
result_vector = feature1 - feature2 + feature3

# generate image using the new calculated vector
result_vector = expand_dims(result_vector, 0)
result_image = model.predict(result_vector)

# scale pixel values for plotting
result_image = (result_image + 1) / 2.0
plt.imshow(result_image[0])
plt.show()