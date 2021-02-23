# https://youtu.be/OQydrlSzxnE
"""

This script downloads images from Google search (or Bing search). 

As with any download, please make sure you are not violating any copyright terms. 
I use this script to download images that help me practice deep learning based
image classification. 

DO NOT use downloaded images to train a commercial product --> this most certainly
violates copyright terms. 

Do not pip install google_images_download

this gives an error that some images could not be downloadable. 
Google changed some things on their side...
The updated repo can be installed using the following command. 
pip install git+https://github.com/Joeclinton1/google-images-download.git

Please remember that this method has a limit of 100 images. 

OR

You can use bing.
Does not seem ot have a limit on the number of images to download. 
pip install bing-image-downloader
"""

from google_images_download import google_images_download

#instantiate the class
response = google_images_download.googleimagesdownload()
arguments = {"keywords":"aeroplane, school bus, dog in front of house",
             "limit":10,"print_urls":False}
paths = response.download(arguments)

#print complete paths to the downloaded images
print(paths)


#####################################
#Bing
from bing_image_downloader import downloader
downloader.download("monkey", limit=200,  output_dir='dataset', 
                    adult_filter_off=True, force_replace=False, timeout=60)
downloader.download("tiger", limit=200,  output_dir='dataset', 
                    adult_filter_off=True, force_replace=False, timeout=60)