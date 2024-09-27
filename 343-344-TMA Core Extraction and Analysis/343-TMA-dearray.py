#https://youtu.be/MHjjHYRFroA
"""
This script extracts individual cores from a Tissue Microarray (TMA) and saves them as PNG images.

It takes a text file with positions as input. This file can be generated using Qupath.

Procedure:
1. Load TMA in Qupath.
2. Select TMA --> TMA De-arrayer --> Run (set rows and columns appropriately).
3. Measure --> Show TMA Measurements --> Save txt file.

Additional Notes:

Before running the script, ensure OpenSlide is installed by executing the following commands:
1. Install openslide-python:
    ```
    pip install openslide-python
    ```
2. Download the latest Windows binaries from https://openslide.org/download/.
3. Extract the contents to a location for easy reference, preferably in the openslide directory in site-packages
On our system: "C:/Users/Admin/Anaconda3/envs/Py39_base/Lib/site-packages/openslide/openslide-win64-20231011/bin"

In case you are unsure of your site packages directory, use the following script to find the location:

import sys
for p in sys.path:
    print(p)
"""

import os
# Specify the path to the OpenSlide DLL directory
openslide_path = "C:/Users/Admin/Anaconda3/envs/Py39_base/Lib/site-packages/openslide/openslide-win64-20231011/bin"
# Add the DLL directory to the system path
os.add_dll_directory(openslide_path)


import cv2
import numpy as np
import openslide
from distutils.util import strtobool
from tqdm.autonotebook import tqdm

# WSI file location
wsi_filename = "your_file_name.svs"
# Text file location - file that contains the coordinates of each core
txt_filename = "core_positions_from_qupath.txt"
# Patch size to extract, around each core
tmaspot_size = 3000
# Directory where PNGs need to be exported
outdir = "core_images/input_images"

print(f"Extracting patches for {wsi_filename} into {outdir}")

# Create output directory if it doesn't exist
if not os.path.isdir(f"{outdir}"):
    os.mkdir(f"{outdir}")

# Use openslide to read the WSI image
slide = openslide.OpenSlide(wsi_filename)

# Set the level for extraction (0 is recommended)
level = 0  # Level=0 --> highest resolution

# Print slide's downsample info
level_dims = slide.level_dimensions[level]
level_downsample = slide.level_downsamples[level]
print(f'Downsample at level {level} is: {level_downsample}')
print(f'WSI dimensions at level {level} are: {level_dims}.')

# Retrieve additional information from slide properties
bounds_x = float(slide.properties.get('openslide.bounds-x', 0))
bounds_y = float(slide.properties.get('openslide.bounds-y', 0))
ratio_x = 1.0 / float(slide.properties['openslide.mpp-x'])
ratio_y = 1.0 / float(slide.properties['openslide.mpp-y'])

# Load coordinates from the text file
dataset = np.loadtxt(txt_filename, dtype=str, skiprows=1)
print(f"Number of rows in txt file: {len(dataset)}")

# Iterate through coordinates and extract images
for row in tqdm(dataset):
    fname, _, label, missing, x, y = row
    if not strtobool(missing):
        # Adjust coordinates using bounds and ratio
        x = (float(x) * ratio_x) + bounds_x
        y = (float(y) * ratio_y) + bounds_y

        print(f"Extracting spot {label} at location", (x, y))

        # Calculate scaled spot size and extract the spot
        scaled_spot_size = (int(tmaspot_size / level_downsample), int(tmaspot_size / level_downsample))
        tmaspot = np.asarray(slide.read_region((int(x - tmaspot_size * 0.5), int(y - tmaspot_size * 0.5)), level,
                                               scaled_spot_size))[:, :, 0:3]
        tmaspot = cv2.cvtColor(tmaspot, cv2.COLOR_RGB2BGR)

        # Save the extracted spot as a PNG image
        cv2.imwrite(f"{outdir}/{label}.png", tmaspot)
    else:
        print(f'The spot {label} is missing, skipping!')

print('Extraction completed for all spots!')
