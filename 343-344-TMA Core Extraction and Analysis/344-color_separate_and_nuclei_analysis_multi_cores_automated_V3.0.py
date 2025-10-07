# https://youtu.be/dF-qnTDJ1-4
"""
Color separate and segment nuclei from multiple images located 
in a directory. 

Save all results in separate directories appropriately labeled inside 
the 'results' directory.

Also, saves the summary of mean brown pixel values from nuclei into a csv file
in the results folder. 
"""

from matplotlib import pyplot as plt
import os
from skimage import io
import pandas as pd
import numpy as np
import time
from color_separate_functions import analyze_TMA_core, color_separate_only, subtract_nuclei_from_brown

# Give path to all cores extracted from TMA
path = "core_images/input_images/"

base_dir_name = os.path.dirname(path).split("/")[0]
results_dir_name = "core_results"
min_nuclei_size = 30
max_nuclei_size = 1000

if not os.path.isdir(base_dir_name + "/" + results_dir_name + "/"):
    os.mkdir(base_dir_name + "/" + results_dir_name + "/")

column_labels = ["File", "meanCytosolR", "StdevCytosolR", "meanCytosolG", "StdevCytosolG", "meanCytosolB", "StdevCytosolB",
                 "meanNucleiR", "StdevNucleiR", "meanNucleiG", "StdevNucleiG", "meanNucleiB", "StdevNucleiB"]
df = pd.DataFrame(columns=column_labels)

for image_name in os.listdir(path):  # iterate through each file to perform some action
    start_time = time.time()
    print("Starting the color separation and nuclei detection process for image - ", image_name)
    
    file_name = image_name.split(".")[0]
    image_path = path + image_name
    
    # Create directory to save results from specific image
    if not os.path.isdir(base_dir_name + "/" + results_dir_name + "/" + file_name + "/"):
        os.mkdir(base_dir_name + "/" + results_dir_name + "/" + file_name + "/")
    
    image = io.imread(image_path)
    orig_image, H, brown, filtered_segm_image, means, stdevs = analyze_TMA_core(image, min_size=min_nuclei_size, max_size=max_nuclei_size)
    
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_original_image.png", orig_image)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_H_Image.png", H)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_Brown_image.png", brown)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_nuclei_image.png", filtered_segm_image, cmap='gray')

    # Subtract nuclei from the brown image and save
    brown_without_nuclei, mean_brown, std_brown, brown_with_nuclei, mean_nuclei, std_nuclei = subtract_nuclei_from_brown(filtered_segm_image, brown)
    
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_Brown_without_nuclei.png", brown_without_nuclei)
    plt.imsave(base_dir_name + "/" + results_dir_name + "/" + file_name + "/" + file_name + "_Brown_with_nuclei.png", brown_with_nuclei)
    
    # Append results to the DataFrame
    temp_dict = pd.DataFrame({"File": [file_name], 
                              "meanCytosolR": [means[0]], 
                              "StdevCytosolR": [stdevs[0]], 
                              "meanCytosolG": [means[1]], 
                              "StdevCytosolG": [stdevs[1]],
                              "meanCytosolB": [means[2]], 
                              "StdevCytosolB": [stdevs[2]],
                              "meanNucleiR": [mean_nuclei[0]], 
                              "StdevNucleiR": [std_nuclei[0]],
                              "meanNucleiG": [mean_nuclei[1]], 
                              "StdevNucleiG": [std_nuclei[1]],
                              "meanNucleiB": [mean_nuclei[2]], 
                              "StdevNucleiB": [std_nuclei[2]]})
    df2 = pd.DataFrame(temp_dict)
                   
    df = pd.concat([df, df2], ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    
    end_time = time.time()
    
    print("Finished analyzing image ", image_name, " in ", (end_time - start_time), " seconds")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

df.to_csv(base_dir_name + "/" + results_dir_name + "/" + "summary_brown_intensity.csv")

