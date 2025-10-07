# https://youtu.be/rFCtZj_r6tA
"""

1. Main Control Function (run_image_retrieval):
   - Takes simple inputs like:
     * What do you want to do? (index new photos or search)
     * Where are your photos?
     * Which photo do you want to find matches for?
   
2. For Indexing:
   - Counts how many photos you have
   - Decides how to organize them efficiently (how many groups to make)
   - Uses the photo organizer (retrieval_system.py) to process all photos
   - Saves the organized system for later use

3. For Searching:
   - Loads your previously organized system
   - Takes your query photo
   - Finds similar photos
   - Shows results in a nice format with:
     * File names
     * Similarity scores
     * Full file paths

4. Smart Features:
   - Automatically adjusts settings based on your photo collection size
   - Handles errors gracefully with helpful messages
   - Lets you customize settings if you want to
   
"""

import os
import logging
import math

# Added to fix OpenMP warning
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

from retrieval_system import ImageRetrievalSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_optimal_regions(num_images: int) -> int:
    """
    Calculate optimal number of regions based on dataset size.
    Rule of thumb: sqrt(N) for small datasets, 4*sqrt(N) for larger ones
    """
    if num_images < 100:
        n_regions = max(1, int(math.sqrt(num_images)))  # At least 1 region
    else:
        n_regions = min(int(4 * math.sqrt(num_images)), num_images // 2)
    return n_regions

def print_results(results):
    """Pretty print the search results."""
    if not results:
        print("\nNo matches found!")
        return
        
    print("\nSearch Results:")
    print("-" * 50)
    for i, (path, distance) in enumerate(results, 1):
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity score
        filename = os.path.basename(path)
        print(f"{i}. Image: {filename}")
        print(f"   Full path: {path}")
        print(f"   Similarity Score: {similarity:.3f}")
        print(f"   Distance: {distance:.3f}")
        print("-" * 50)

def run_image_retrieval(
    task: str = "index",              
    image_dir: str = None,            
    query_image: str = None,          
    index_path: str = "image_index.faiss",
    metadata_path: str = "image_metadata.json",
    num_results: int = 5,
    n_regions: int = None,            
    nprobe: int = None,               
    use_gpu: bool = False
) -> None:
    """
    Run image retrieval system in either index or search mode.
    """
    try:
        if task.lower() == 'index':
            if not image_dir:
                raise ValueError("image_dir is required for indexing task")
            
            # Count number of images in directory
            image_files = [f for f in os.listdir(image_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            num_images = len(image_files)
            
            # Calculate optimal number of regions if not specified
            if n_regions is None:
                n_regions = calculate_optimal_regions(num_images)
            if nprobe is None:
                nprobe = max(1, n_regions // 4)
                
            logger.info(f"Number of images: {num_images}")
            logger.info(f"Using {n_regions} regions and searching {nprobe} regions")
                
            retrieval_system = ImageRetrievalSystem(
                n_regions=n_regions,
                nprobe=nprobe,
                use_gpu=use_gpu
            )
            
            logger.info(f"Indexing images from {image_dir}")
            retrieval_system.index_images(image_dir=image_dir)
            
            logger.info("Saving index and metadata")
            retrieval_system.save(index_path, metadata_path)
            
        elif task.lower() == 'search':
            if not query_image:
                raise ValueError("query_image is required for search task")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                raise ValueError(f"Index or metadata file not found. Please ensure both exist:\n"
                               f"Index: {index_path}\nMetadata: {metadata_path}")
                
            logger.info(f"Loading existing index for search")
            retrieval_system = ImageRetrievalSystem(
                index_path=index_path,
                metadata_path=metadata_path,
                nprobe=nprobe if nprobe is not None else 10,  # Use default if not specified
                use_gpu=use_gpu
            )
            
            logger.info(f"Searching for similar images to {query_image}")
            results = retrieval_system.search(
                query_image_path=query_image,
                k=num_results
            )
            
            print_results(results)
            
        else:
            raise ValueError("Task must be either 'index' or 'search'")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Parameters
    TASK = "search"  # Change this to "index" or "search" for searching
    IMAGE_DIR = "all_images"  # Directory containing images to index
    QUERY_IMAGE = "query_images/tiger3.jpg"  # Image to search for
    INDEX_PATH = "image_index.faiss"   # Where to save/load the index
    METADATA_PATH = "image_metadata.json"  # Where to save/load the metadata
    
    # Example for indexing images
    if TASK == "index":
        run_image_retrieval(
            task="index",
            image_dir=IMAGE_DIR,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            use_gpu=False
        )
    
    # Example for searching
    elif TASK == "search":
        run_image_retrieval(
            task="search",
            query_image=QUERY_IMAGE,
            index_path=INDEX_PATH,
            metadata_path=METADATA_PATH,
            num_results=5,
            nprobe=10,  # Optional: specify number of regions to search
            use_gpu=False
        )