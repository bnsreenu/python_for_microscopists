# https://youtu.be/rFCtZj_r6tA
"""
Image Retrieval System

This code implements a system for finding similar images using feature-based similarity search. 
It extracts visual features from images using a neural network and enables fast similarity 
search through the following main components:

1. Feature Extraction: Converts images into numerical feature vectors that capture their 
   visual characteristics (handled by a separate ImageFeatureExtractor class)

2. Indexing: 
   - Processes a directory of images and extracts their features
   - Stores these features in a FAISS index (Facebook AI Similarity Search) 
   - Maintains metadata about each indexed image (path, filename, indexing date)

3. Search: 
   - Takes a query image and finds the k most similar images from the indexed collection
   - Uses IndexIVFFlat to measure similarity between images
   - Returns matched images sorted by similarity score

Note about IndexIVFFlat:
    - Uses a "divide and conquer" approach
    - First divides vectors into clusters/regions
    - When searching:
      * First finds which clusters are most relevant
      * Only searches within those chosen clusters
    - Requires two extra steps:
      * Training: Learning how to divide vectors into clusters
      * nprobe: Choosing how many clusters to check (tradeoff between speed and accuracy)
    - Usually much faster for large datasets
    - Might miss some matches (approximate search) but usually good enough

"""



import os
import json
import torch
import faiss
import numpy as np
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
from datetime import datetime
import logging
from feature_extractor import ImageFeatureExtractor, ImageDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageRetrievalSystem:
    def __init__(self, 
                 feature_extractor: Optional[ImageFeatureExtractor] = None,
                 index_path: Optional[str] = None,
                 metadata_path: Optional[str] = None,
                 use_gpu: bool = False,
                 n_regions: int = 100,  
                 nprobe: int = 10):    
        """Initialize the retrieval system with IVF index."""
        self.feature_extractor = feature_extractor or ImageFeatureExtractor()
        self.feature_dim = self.feature_extractor.feature_dim
        self.n_regions = n_regions
        self.nprobe = nprobe
        logger.info(f"Initializing retrieval system with dimension: {self.feature_dim}")
        
        self.metadata = {}
        self.is_trained = False
        
        # Load existing index and metadata if provided
        if index_path and metadata_path:
            self.load(index_path, metadata_path)
        else:
            # Initialize new FAISS IVF index
            logger.info(f"Creating new IVF index with {n_regions} regions")
            self.quantizer = faiss.IndexFlatL2(self.feature_dim)
            self.index = faiss.IndexIVFFlat(self.quantizer, self.feature_dim, 
                                          self.n_regions, faiss.METRIC_L2)
            self.index.nprobe = self.nprobe
            
            # Convert to GPU index if requested
            if use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Successfully moved index to GPU")
                except Exception as e:
                    logger.warning(f"Failed to use GPU, falling back to CPU: {str(e)}")



    def index_images(self, 
                    image_dir: str, 
                    batch_size: int = 32,
                    num_workers: int = 4) -> None:
        """Index all images in the specified directory."""
        logger.info(f"Indexing images from {image_dir}")
        
        # Get all image paths
        image_paths = [
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
        ]
        
        features_list = []
        valid_paths = []
        
        # Process images one by one
        for img_path in image_paths:
            try:
                features = self.feature_extractor.extract_features(img_path)
                features_list.append(features)
                valid_paths.append(img_path)
                logger.info(f"Processed {img_path}")
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
        
        if not features_list:
            raise ValueError("No valid features extracted from images")
            
        # Combine all features
        all_features = np.stack(features_list)
        logger.info(f"Feature array shape: {all_features.shape}")
        logger.info(f"Feature stats - Min: {all_features.min():.4f}, Max: {all_features.max():.4f}")
        
        # Train index if not already trained
        if not self.is_trained:
            logger.info("Training IVF index...")
            self.index.train(all_features)
            self.is_trained = True
            logger.info("Index training completed")
        
        # Add to index
        self.index.add(all_features)
        logger.info(f"Total vectors in index: {self.index.ntotal}")
        
        # Update metadata
        for idx, path in enumerate(valid_paths):
            self.metadata[str(idx)] = {
                'path': path,
                'filename': os.path.basename(path),
                'indexed_at': datetime.now().isoformat()
            }
        
        logger.info(f"Successfully indexed {len(valid_paths)} images")

    def search(self, 
              query_image_path: str,
              k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar images."""
        logger.info(f"Searching for similar images to {query_image_path}")
        logger.info(f"Total images in index: {self.index.ntotal}")
        logger.info(f"Available metadata keys: {list(self.metadata.keys())}")
        
        if not self.is_trained:
            raise RuntimeError("Index has not been trained. Add images first.")
        
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image_path)
        logger.info(f"Query feature shape: {query_features.shape}")
        
        # Search index
        k = min(k, self.index.ntotal)  # Make sure k doesn't exceed number of indexed images
        distances, indices = self.index.search(
            query_features.reshape(1, -1),
            k
        )
        
        logger.info(f"Raw search results - distances: {distances[0]}")
        logger.info(f"Raw search results - indices: {indices[0]}")
        logger.info(f"Searched {self.nprobe} out of {self.n_regions} regions")
        
        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            str_idx = str(int(idx))
            if str_idx in self.metadata:
                results.append((self.metadata[str_idx]['path'], float(dist)))
                logger.info(f"Match found: {self.metadata[str_idx]['path']} with distance {dist:.3f}")
            else:
                logger.warning(f"Index {idx} not found in metadata")
        
        # Sort results by distance (smaller is better)
        results.sort(key=lambda x: x[1])
        
        if not results:
            logger.warning("No matches found!")
        else:
            logger.info(f"Found {len(results)} matches")
            
        return results

    def save(self, index_path: str, metadata_path: str) -> None:
        """Save the index and metadata to disk."""
        # If index is on GPU, convert back to CPU for saving
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_gpu_to_cpu(self.index)
            
        faiss.write_index(self.index, index_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f)
            
        logger.info(f"Saved index with {self.index.ntotal} vectors")
        logger.info(f"Saved index to {index_path} and metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str) -> None:
        """Load the index and metadata from disk."""
        logger.info(f"Loading index from {index_path}")
        self.index = faiss.read_index(index_path)
        self.is_trained = True  # Loaded indexes are already trained
        
        # Set nprobe for loaded index
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = self.nprobe
            logger.info(f"Set nprobe to {self.nprobe} for loaded IVF index")
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        logger.info(f"Loaded index with {self.index.ntotal} vectors")
        logger.info(f"Metadata contains {len(self.metadata)} entries")