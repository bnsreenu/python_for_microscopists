# https://youtu.be/qMN7YmpnzHE

import os
import shutil
import json


# Path to your original coco JSON annotation files for test and val
test_annotation_path = 'livecell_train_val_images/livecell_coco_val.json'

# Directory containing all the images
images_directory = 'livecell_train_val_images'

# Output directories for test subset
output_test_dir = 'SHSY5Y-images/val2'

# Create output directories if they don't exist
os.makedirs(output_test_dir, exist_ok=True)

# List of image filenames to select for the test subset
test_image_filenames = []

# Define a function to copy images from the original dataset to the subsets
def copy_images(filenames, source_dir, dest_dir):
    for i, filename in enumerate(filenames, start=1):
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.copyfile(src_path, dest_path)
        print(f"Copying image {i}/{len(filenames)}: {filename}")

# Load original annotations
with open(test_annotation_path, 'r') as test_annotation_file:
    test_annotations = json.load(test_annotation_file)


# Filter images based on filenames for the test subset
for image_info in test_annotations['images']:
    filename = image_info['file_name']
    if filename.startswith("SHSY5Y"):
        test_image_filenames.append(filename)


# Copy images to the test and val subsets
print("Copying test images...")
copy_images(test_image_filenames, source_dir=images_directory, dest_dir=output_test_dir)

# Create dictionaries for filtered annotations
filtered_test_annotations = {
    "images": [],
    "annotations": [],
    "categories": test_annotations['categories']  # You may need to include category information if applicable
}


# Filter annotations for the selected images
for image_info in test_annotations['images']:
    if image_info['file_name'] in test_image_filenames:
        filtered_test_annotations['images'].append(image_info)
        annotations = [ann for ann in test_annotations['annotations'] if ann['image_id'] == image_info['id']]
        filtered_test_annotations['annotations'].extend(annotations)


# Save the filtered annotations for test and val
filtered_test_json_path = os.path.join(output_test_dir, "filtered_test_annotations.json")

print("Writing json for test images...")
with open(filtered_test_json_path, "w") as filtered_test_json_file:
    json.dump(filtered_test_annotations, filtered_test_json_file, indent=4)


print(f"Subset images and JSON annotations created in {output_test_dir} ")






