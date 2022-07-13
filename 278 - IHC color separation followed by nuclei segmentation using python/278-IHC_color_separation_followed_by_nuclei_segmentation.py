# https://youtu.be/7u83l_HqamI
"""
RGB to Haematoxylin-Eosin-DAB (HED) color space conversion followed by nuclei segmentation
and analysis

separate the immunohistochemical (IHC) staining from the hematoxylin counterstaining
The IHC staining expression of the FHL2 protein is here revealed with diaminobenzidine (DAB) which gives a brown color.

A. C. Ruifrok and D. A. Johnston, “Quantification of histochemical staining by color deconvolution,” Analytical and quantitative cytology and histology / the International Academy of Cytology [and] American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001. PMID: 11531144

https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_ihc_color_separation.html#sphx-glr-auto-examples-color-exposure-plot-ihc-color-separation-py

Try WSI datasets from here
https://zenodo.org/record/1485967#.Yd31lv7MKbh
https://www.wouterbulten.nl/blog/tech/peso-dataset-whole-slide-image-prosate-cancer/

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb
from skimage.exposure import rescale_intensity

# Separate the individual stains from the IHC image
def color_separate(ihc_rgb):

    #Convert the RGB image to HED using the prebuilt skimage method
    ihc_hed = rgb2hed(ihc_rgb)
    
    # Create an RGB image for each of the separated stains
    #Convert them to ubyte for easy saving to drive as an image
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = img_as_ubyte(hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1)))
    ihc_e = img_as_ubyte(hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1)))
    ihc_d = img_as_ubyte(hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1)))

    #Optional fun exercise of combining H and DAB stains into a single image with fluorescence look
    
    h = rescale_intensity(ihc_hed[:, :, 0], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 0], 99)))
    d = rescale_intensity(ihc_hed[:, :, 2], out_range=(0, 1),
                          in_range=(0, np.percentile(ihc_hed[:, :, 2], 99)))

# Cast the two channels into an RGB image, as the blue and green channels
#Convert to ubyte for easy saving as image to local drive
    zdh = img_as_ubyte(np.dstack((null, d, h))) #DAB in green and H in Blue

    return (ihc_h, ihc_e, ihc_d, zdh)


ihc_rgb =io.imread("images/Ki-67.jpg")

plt.imshow(ihc_rgb)
plt.axis("off")

H,E,D,HD = color_separate(ihc_rgb)
plt.imsave('images/H_img.jpg', H)
plt.imsave('images/DAB_img.jpg', D)

plt.imshow(H)
plt.axis("off")

################################################
#Segmentation using Voronoi-Otsu labeling
# For installation instructions of the package, please refer to the following link
# https://github.com/clEsperanto/pyclesperanto_prototype
##########################


import pyclesperanto_prototype as cle

# select a specific OpenCL / GPU device and see which one was chosen
device = cle.select_device('RTX')
print("Used GPU: ", device)

input_image = np.invert(D[:,:,2])
plt.imshow(input_image, cmap='gray')
#Before segmenting the image, need to push it to GPU memory. For visualisation purposes we crop out a sub-region:
input_gpu = cle.push(input_image)


cle.imshow(input_gpu)
    
sigma_spot_detection = 3
sigma_outline = 1

segmented = cle.voronoi_otsu_labeling(input_gpu, spot_sigma=sigma_spot_detection, 
                                      outline_sigma=sigma_outline)

cle.imshow(segmented, labels=True)

   
statistics = cle.statistics_of_labelled_pixels(input_gpu, segmented) 

import pandas as pd
table = pd.DataFrame(statistics)    

print(table.describe())
print(table.info())
