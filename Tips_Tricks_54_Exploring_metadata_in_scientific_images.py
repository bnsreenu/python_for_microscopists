# https://youtu.be/A4po9z61TME

# ALL ABOUT METADATA IN MEDICAL AND SCIENTIFIC IMAGES

"""
What is Metadata?
It is the additional information about a file, providing details such as the 
creation date, author, location, pixel size, experimental settings, etc. 

Why is Metadata important? 
For your travel images, it is important so you know when and where the image was taken. 
Please note that metadata is not necessary but useful when some information is
needed at a future date. May be your grandkids want to take a picture in future 
at the same location on your 100th birthday!

For scientific images, it is important to ensure traceability, interoperability,
and reproducibility. 

Metadata provides a detailed history of the image, including acquisition parameters, 
equipment settings, and processing steps. This traceability is crucial for 
tracking the origin and evolution of the data, ensuring accountability and 
transparency in scientific research.

Standardized metadata formats enable different software and systems to understand 
and interpret information consistently. This promotes interoperability, allowing 
researchers to share and collaborate on data across various platforms and tools 
without losing critical details.

Metadata contains essential information about the experimental setup and conditions. 
Reproducing scientific experiments requires accurate knowledge of these factors. 
With comprehensive metadata, other researchers can precisely replicate experiments, 
verify results, and build upon existing work, contributing to the reliability 
and credibility of scientific findings.

Images come in many formats, let us explore metadata from a few most-common 
image formats including JPG, DICOM, TIFF, GEO-TIFF, OME-TIFF, and .CZI

"""
###########################################################################
## JPEG/JPG IMAGES

"""
JPEG images - from our phones and cameras

Metadata for JPG images is standardized, and the standard used is called 
Exif (Exchangeable image file format).
JPG typically stores basic metadata like date, camera settings, and location in 
the Exif header.

Japan Electronics and Information Technology Industries Association (JEITA) produced
the initial dfinition of Exif standard. Today, JEITA along with  
Camera & Imaging Products Association (CIPA) defines and maintains 
the Exif standard to ensure compatibility and uniformity in metadata across 
various devices and software that handle digital images.

"""

from PIL import Image

# Open the image
image = Image.open('tower_bridge.jpg')

# Extract the EXIF metadata
exif_data = image.getexif()

# Print the EXIF metadata
for tag, value in exif_data.items():
    print(f"{tag}: {value}")     


import PIL.ExifTags
exif = {
    PIL.ExifTags.TAGS[k]: v
    for k, v in image._getexif().items()
    if k in PIL.ExifTags.TAGS
}

#########################################################

### MOVING ON TO SCIENIFIC IMAGES ##################

##########################################################

"""
## DICOM images ###

DICOM (Digital Imaging and Communications in Medicine) is the international 
standard for medical images. DICOM metadata standards were originally defined by the 
American College of Radiology (ACR) and the National Electrical Manufacturers 
Association (NEMA) in 1983 by forming a joint committee. 
Today, the DICOM Standard is managed by the 
Medical Imaging & Technology Alliance - a division of NEMA.

Metadata in DICOM files includes information about the image data, such as size, 
dimensions, bit depth, and modality used to create the data. DICOM metadata 
also includes patient information, image position data, and other relevant details.

pip install pydicom
# https://github.com/pydicom/pydicom
# Sample DICOM images downloaded from here: https://www.rubomedical.com/dicom_files/

#Accessing metadata using their tags
# https://www.dicomlibrary.com/dicom/dicom-tags/

"""
from matplotlib import pyplot as plt
from pydicom import dcmread

#  Read the DICOM file
dicom_data = dcmread('brain.DCM')

# Access and print the metadata
print("DICOM Metadata:")
print(f"Patient Name: {dicom_data.PatientName}")
print(f"Patient ID: {dicom_data.PatientID}")
print(f"Modality: {dicom_data.Modality}")
print(f"Image Type: {dicom_data.ImageType}")

#Accessing metadata using their tags
# https://www.dicomlibrary.com/dicom/dicom-tags/

# Access the image type metadata using the tag (0008, 0008)
image_type = dicom_data[(0x0008, 0x0008)].value
# Print the image type metadata
print("Image Type:", image_type)

physician_name = dicom_data[(0x0008, 0x0090)].value
print("Physician's Name':", physician_name)

image_array = dicom_data.pixel_array
plt.imshow(image_array, cmap='gray')


## Let us enhance the image for visualization by normalizing pixel values 
# between 0 and 1 and the performing histogram equalization
import numpy as np
import cv2
# Normalize the pixel values between 0 and 1
pixel_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))

# Apply histogram equalization
pixel_array = np.uint8(255 * pixel_array)  # Convert back to uint8 for histogram equalization
pixel_array_eq = cv2.equalizeHist(pixel_array)

# Display the image
plt.imshow(pixel_array_eq, cmap='gray')
plt.axis('off')
plt.show()

######################################################

"""
TIFF images: 
    
The TIFF format was initially introduced by Aldus Corporation, and the 
first version, TIFF 1.0, was released in 1986. Adobe Systems later became 
involved in the development and standardization of TIFF. Today, Adobe holds the 
copyright on the TIFF specification. 
    

Tiff uses a structured format called the Image File Directory (IFD) to store 
metadata information. The IFD contains tags that define various metadata fields 
such as image dimensions, color space, compression method, and more. 
Each tag in the IFD has a unique identifier (TagID) and is associated with a 
specific metadata field. The raw data for each tag is stored in the TIFF file 
along with its TagType, Count, and Offset. 
    
Useful resources:
https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml

"""

# First drag a tif image to a notepad to read some readable metadata content. 
# Then, try the following code to further extend your understanding. 


import tifffile

######## Print all metadata associated with an image ######
# Open the TIFF file
with tifffile.TiffFile('mito.tif') as tiff:
    # Get the metadata
    metadata = tiff.pages[0].tags

    # Print all the metadata fields
    for tag in metadata.values():
        print(tag.name, tag.value)


### Now, let us extract some metadata and print ####
#First let us see what type of informationis stored in the tiff file. 
#Print all attributes stored as part of the tiff_image.pages[0] attribute
tiff_image = tifffile.TiffFile('mito-0.2um_pixel.tif')  
# Also try mito-0.2um_pixel.tif and Ti64.10X.tif
# mito-0.2um_pixel.tif has been modified in imageJ. 
attributes = dir(tiff_image.pages[0])
print(attributes)

#If the image has been modified in imageJ then imagej_metadata exists
print("Does this image have imageJ metadata? ", tiff_image.is_imagej)
print(tiff_image.imagej_metadata)
# Extract the pixel size information
pixel_width = tiff_image.imagej_metadata.get("spacing_x", 1.0)  # Default to 1.0 if not found
pixel_height = tiff_image.imagej_metadata.get("spacing_y", 1.0)
print("Pixel spacing in x and y is: ", pixel_width, pixel_height)


#We can also use specific information using its respective tag identifier
# https://www.loc.gov/preservation/digital/formats/content/tiff_tags.shtml


#Let us extract and print some metadata
def extract_tiff_metadata(tiff_path):
    # Open the TIFF image
    tiff_image = tifffile.TiffFile(tiff_path)

    # Extract the image dimensions
    width, height, channels = tiff_image.pages[0].shape  #Shape depends on the image type

    # Extract the color space information
    color_space = tiff_image.pages[0].photometric

    # Extract the bit depth
    bit_depth = tiff_image.pages[0].bitspersample

    # Extract the compression method
    compression = tiff_image.pages[0].compression

    # Extract the image acquisition date and time
    acquisition_date = tiff_image.pages[0].tags.get(306)

    # Extract the image description
    description = tiff_image.pages[0].description

    # Extract the software used
    software = tiff_image.pages[0].software

    # Extract the resolution
    resolution = tiff_image.pages[0].tags.get(282)

    # Extract the camera or device information
    device_info = tiff_image.pages[0].tags.get(271)

    # Extract the GPS coordinates
    gps_info = tiff_image.pages[0].tags.get(34853)

    # Close the TIFF image
    tiff_image.close()

    # Return the extracted metadata
    metadata = {
        'width': width,
        'height': height,
        'color_space': color_space,
        'bit_depth': bit_depth,
        'compression': compression,
        'acquisition_date': acquisition_date,
        'description': description,
        'software': software,
        'resolution': resolution,
        'device_info': device_info,
        'gps_info': gps_info
    }
    return metadata

# Example usage
#tiff_path = 'mito.tif'
tiff_path = 'Ti64.10X.tif'
metadata = extract_tiff_metadata(tiff_path)

# Print the extracted metadata
for key, value in metadata.items():
    print(f'{key}: {value}')
    
#####################################################

"""
Geotiff:
    
GeoTIFF is an extension of the TIFF file format that adds geospatial metadata 
to the image. It allows for the storage of geographic information along with 
the raster data. The metadata in GeoTIFF is stored in the form of tags within 
the TIFF file. 

The GeoTIFF standard is maintained by the Open Geospatial Consortium (OGC)
    
Download and install GDAL first
https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
Since I have python 3.7 I downloaded: GDAL-3.1.4-cp37-cp37m-win_amd64.whl
cp37 stands for python3.7
You can just install these using pip

Download and install rasterio
https://www.lfd.uci.edu/~gohlke/pythonlibs/#rasterio
Since I have python 3.7 I downloaded: rasterio-1.1.8-cp37-cp37m-win_amd64.whl
cp37 stands for python3.7

Images from: https://download.osgeo.org/geotiff/samples/usgs/

"""

import rasterio

from rasterio.plot import show
from matplotlib import pyplot as plt


img = rasterio.open('c41078a1.tif')
show(img)

#To find out number of bands in an image
num_bands = img.count
print("Number of bands in the image = ", num_bands)

img_band1 = img.read(1) #1 stands for 1st band. 

band_number=1
with rasterio.open('c41078a1.tif') as src:
    print(src.tags(band_number))


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

#################
"""
## OMETIFF FILES

OME-TIFF (Open Microscopy Environment - Tagged Image File Format) is a file 
format designed for microscopy images and metadata. It is an extension of the 
TIFF file format and is part of the broader Open Microscopy Environment (OME) 
initiative. 

OME-TIFF aims to provide a standardized way to store multidimensional microscopy 
data and associated metadata. OME-TIFF stores metadata in a structured way using 
XML embedded within the TIFF file. The XML contains information about the 
microscope settings, acquisition parameters, and other meatadata. 

The metadata in OME-TIFF sticks to the OME Data Model, which defines a standard 
way of representing microscopy-related information.OME Consortium is responsible 
for the development and maintenance of the OME-TIFF standard. 
The OME Consortium is a collaborative effort involving multiple organizations 
and institutions with an interest in open standards for microscopy data.

https://docs.openmicroscopy.org/ome-model/6.0.1/

When it comes to python: ometiff files can be read using tifffile library and 
metadata extracted the same way as above, like in regular tiff.
But, let us use apeer_ometiff_library to extract the embedded xml metadata file.

Reading OME-TIFF using apeer_ometiff_library 
# pip install apeer-ometiff-library 
# to import the package you need to use import apeer_ometiff_library
#OME-TIFF has tiff and metada (as XML) embedded
#Image is a 5D array.

"""

# First, let us use our tiffifle library from the above exercise. 
import tifffile
ome_tiff_image = tifffile.TiffFile('Osteosarcoma_01.ome.tiff')  
attributes = dir(ome_tiff_image.pages[0])
print(attributes)

#Let us use a different library. (apeer-ometiff-library)
from apeer_ometiff_library import io  

(img, omexml) = io.read_ometiff("Osteosarcoma_01.ome.tiff")  #Unwrap image and embedded xml metadata
print (img.shape)   #to verify the shape of the array
print(img)

print(omexml)


# Let us have a look at the image ...
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the first subplot
axes[0,0].imshow(img[0,0,0,:,:], cmap='gray')
axes[0,0].axis('off')

# Plot the second image on the second subplot
axes[0,1].imshow(img[0,1,0,:,:], cmap='gray')
axes[0,1].axis('off')

# Plot the third image on the third subplot
axes[1,0].imshow(img[0,2,0,:,:], cmap='gray')
axes[1,0].axis('off')

axes[1, 1].axis('off')

# Adjust the layout
fig.tight_layout()

# Show the plot
plt.show()



#########################
"""
# .czi files

Microscope manufacturers often develop proprietary file formats like .czi for 
many reasons. 

Instrument specific metadata: This includes details like laser settings, 
filter configurations, stage positions, and other instrument-specific parameters.

Efficient Data Storage: Proprietary formats may be optimized for efficient 
data storage, which can be crucial during high-throughput imaging where large 
amounts of data are generated rapidly. These formats may be tailored to the 
specific requirements of the imaging system, potentially allowing for faster 
data acquisition and writing to storage.

Instrument Control and Integration: Proprietary formats are often tightly 
integrated with the manufacturer's instrument control software. This integration 
can enable seamless communication between the microscope hardware and software, 
ensuring accurate representation of experimental parameters and facilitating 
real-time adjustments during acquisition.

Advanced Features and Compression: Some proprietary formats may support advanced 
features specific to certain microscope models, such as multidimensional data, 
time-series imaging, or specialized compression algorithms. These features can 
be optimized for the unique capabilities of the microscope, potentially providing 
advantages in terms of speed and efficiency.

Market Differentiation: Manufacturers may use proprietary formats as a means of 
product differentiation. Having a unique format can encourage users to stick 
with the manufacturer's ecosystem for data analysis and processing, fostering 
brand loyalty.

Most vendors provide a way to save data into ohter formats, including ometiff.

Let us have a look at a .CZI file:
    
These files include a lot of metadata, making almost every aspect of image acquisition
available to the user. Key metadata can be easily accessed using ZEISS' ZEN software. 
A free version is available to open .czi files (ZEN Lite). Of course, imageJ can
be used to read metadata from any file (Image --> Show Info.)

ZEISS' python API (pylibCZIrw) allows for reading and writing .czi files including
access to the extensive metadata.
pip install pylibCZIrw

To learn more about working with .czi files in python:
https://colab.research.google.com/github/zeiss-microscopy/OAD/blob/master/jupyter_notebooks/pylibCZIrw/pylibCZIrw_3_3_0.ipynb

"""

from pylibCZIrw import czi as pyczi
import json

input_image_path = 'Osteosarcoma_01.czi'

# open the CZI for reading using a context manager (preferred way to do it)
# and print the xml metadata
with pyczi.open_czi(input_image_path) as czidoc:
    # get the raw metadata as XML
    md_xml = czidoc.raw_metadata
    print(md_xml)


# Instead of sifting through the extensive metadata, let us print what we need.
# In this case, print the information related to all Channels
#Then, print the dye name for Channel 0
with pyczi.open_czi(input_image_path) as czidoc:
    # get the raw metadata as a dictionary
    md_dict = czidoc.metadata

    # Print something specific, like the channel information
    print(json.dumps(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"], sort_keys=False, indent=4))
    print("__________________________________________")
    print("Dye Name for channel 0: ", json.dumps(md_dict["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"][0]["DyeName"], sort_keys=False, indent=4))

    

# Let us print more metadata
with pyczi.open_czi(input_image_path) as czidoc:
    # get the image dimensions as an dictionary, where the key identifies the dimension
    total_bounding_box = czidoc.total_bounding_box
    print("Image dimensions are: ", total_bounding_box)
    # get the pixel type for all channels
    pixel_types = czidoc.pixel_types
    print("The pixel types for all channels are: ", pixel_types)


# Let us have a look at the image corresponding to the above metadata/ 
#To read a plane from the czi file
with pyczi.open_czi(input_image_path) as czidoc:
    # define some plane coordinates
    plane_0 = {'C': 0, 'Z': 0, 'T': 0}
    plane_1 = {'C': 1, 'Z': 0, 'T': 0}
    plane_2 = {'C': 2, 'Z': 0, 'T': 0}

    channel_0 = czidoc.read(plane=plane_0)
    channel_1 = czidoc.read(plane=plane_1)
    channel_2 = czidoc.read(plane=plane_2)


fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Plot the first image on the first subplot
axes[0,0].imshow(channel_0[:,:,0], cmap='gray')
axes[0,0].axis('off')

# Plot the second image on the second subplot
axes[0,1].imshow(channel_1[:,:,0], cmap='gray')
axes[0,1].axis('off')

# Plot the third image on the third subplot
axes[1,0].imshow(channel_2[:,:,0], cmap='gray')
axes[1,0].axis('off')

axes[1, 1].axis('off')

# Adjust the layout
fig.tight_layout()

# Show the plot
plt.show()


############################################
