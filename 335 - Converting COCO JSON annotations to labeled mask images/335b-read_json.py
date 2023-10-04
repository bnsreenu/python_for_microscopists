# https://youtu.be/SQng3eIEw-k
"""

How to read a COCO JSON file to understand the structure of the data?

"""

import json

# Load the JSON file
with open('livecell_test_images/livecell_coco_test.json', 'r') as file:
    data = json.load(file)

# Print out the first few image filenames/paths
for img in data['images'][:10]:  # Adjust the number to print more or fewer paths
    print(img['file_name'])


################################

#Understanding the structure of JSON:
    
import json

def print_structure(d, indent=0):
    """Print the structure of a dictionary or list."""
    
    # If the input is a dictionary
    if isinstance(d, dict):
        for key, value in d.items():
            print('  ' * indent + str(key))
            print_structure(value, indent+1)
            
    # If the input is a list
    elif isinstance(d, list):
        print('  ' * indent + "[List of length {} containing:]".format(len(d)))
        if d:
            print_structure(d[0], indent+1)  # Only print the structure of the first item for brevity

# Load the JSON file
with open('livecell_test_images/livecell_coco_test.json', 'r') as file:
    data = json.load(file)

print_structure(data)
