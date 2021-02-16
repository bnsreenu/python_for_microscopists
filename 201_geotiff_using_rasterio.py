# https://youtu.be/ieyODuIjXp4
"""
@author: Sreenivas Bhattiprolu

Download and install GDAL first
https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
Since I have python 3.7 I downloaded: GDAL-3.1.4-cp37-cp37m-win_amd64.whl
cp37 stands for python3.7

Download and install rasterio
https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
Since I have python 3.7 I downloaded: rasterio-1.1.8-cp37-cp37m-win_amd64.whl
cp37 stands for python3.7

Images from: https://landsatonaws.com/L8/042/034/LC08_L1TP_042034_20180619_20180703_01_T1



"""

import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt

img = rasterio.open('images/MOS_CZ_KR_250.tif')
show(img)
#X and Y are supposed to be latitude and longitude if you have the right metadata

full_img = img.read()  #Note the 3 bands and shape of image


#To find out number of bands in an image
num_bands = img.count
print("Number of bands in the image = ", num_bands)

img_band1 = img.read(1) #1 stands for 1st band. 
img_band2 = img.read(2) #2 stands for 2nd band. 
img_band3 = img.read(3) #3 stands for 3rd band. 

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img_band1, cmap='pink')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(img_band2, cmap='pink')
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(img_band3, cmap='pink')

#To find out the coordinate reference system
print("Coordinate reference system: ", img.crs)

# Read metadata
metadata = img.meta
print('Metadata: {metadata}\n'.format(metadata=metadata))

#Read description, if any
desc = img.descriptions
print('Raster description: {desc}\n'.format(desc=desc))


#To find out geo transform
print("Geotransform : ", img.transform)

## Plot pixel value histogram in each band. 
rasterio.plot.show_hist(full_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)
# Peak at 255 is pixels with no data, outside region of interest.

clipped_img = full_img[:, 300:900, 300:900]
plt.imshow(clipped_img[0,:,:])
rasterio.plot.show_hist(clipped_img, bins=50, histtype='stepfilled', lw=0.0, stacked=False, alpha=0.3)
#Each band showing slightly different information

################# NDVI - normalized difference vegetation index ############
# NDVI = (NIR-Red)/(NIR+Red)

#Let us assume 1 is red and 2 is NIR
red_clipped = clipped_img[0].astype('f4')
nir_clipped = clipped_img[1].astype('f4')
ndvi_clipped = (nir_clipped - red_clipped) / (nir_clipped + red_clipped)

# Return Runtime warning about dividing by zero as we have some pixels with value 0.
# So let us use numpy to do this math and replace inf / nan with some value. 

import numpy as np
ndvi_clipped2 = np.divide(np.subtract(nir_clipped, red_clipped), np.add(nir_clipped, red_clipped))
ndvi_clipped3 = np.nan_to_num(ndvi_clipped2, nan=-1)
plt.imshow(ndvi_clipped3, cmap='viridis')
plt.colorbar()
#Some times each band is available as seperate images
#Data from here: https://landsatonaws.com/L8/042/034/LC08_L1TP_042034_20180619_20180703_01_T1
#Band 4 = Red, Band 5: NIR

red = rasterio.open('images/landsat/red_band.tif')
#Extract image as a smaller size... 
red_img = red.read(1, out_shape=(1, int(red.height // 2), int(red.width // 2)))
plt.imshow(red_img, cmap='viridis')
plt.colorbar()
#Extract smaller region, otherwise when we do NDVI math we divide by 0 where there is no data
red_img = red_img[1000:3000, 1000:3000]
plt.imshow(red_img, cmap='viridis')
plt.colorbar()

nir = rasterio.open('images/landsat/NIR_band.tif')
nir_img = nir.read(1, out_shape=(1, int(nir.height // 2), int(nir.width // 2)))
nir_img = nir_img[1000:3000, 1000:3000]

plt.imshow(nir_img, cmap='viridis')
plt.colorbar()

#Convert int to float as we will be doing math
red_img_float = red_img.astype('f4') #Float 32
nir_img_float = nir_img.astype('f4')

ndvi = (nir_img_float - red_img_float) / (nir_img_float + red_img_float)
plt.imshow(ndvi, cmap='viridis')
plt.colorbar()

