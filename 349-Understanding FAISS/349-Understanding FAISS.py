# https://youtu.be/0jOlZpFFxCE
"""
Cosine Similarity Explained:
    
Cosine similarity compares how "aligned" two vectors are (like comparing directions they point)
We compare query vector against all database vectors
Closest alignments (highest cosine similarity) are the best matches


Measures the cosine of the angle between two vectors
Formula: cos(θ) = (A·B)/(||A||·||B||)
Result ranges from -1 (opposite) to 1 (identical)
Value of 0 indicates perpendicularity (no similarity)


FAISS Search Types Explained:

Faiss is a library for efficient similarity search and clustering of dense vectors. 
Faiss contains several methods for similarity search.

Two main approaches:

IndexFlatL2: Exact L2 matching but faster than manual implementation
IndexIVF: Clusters similar features together, only searches relevant clusters


1. FAISS Flat L2 (IndexFlatL2):
   - The simplest, most accurate search method
   - Checks every single vector in the database ("brute force" within FAISS)
   - Always finds the exact nearest neighbors
   - Fast for small datasets, but gets slower as data grows
   - No training required - just add vectors and search

2. FAISS IVF (IndexIVFFlat - Inverted File):
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



When to use which:
- Use Cosine OR Flat L2 when:
  * Dataset is small (thousands of vectors)
  * Need exact results
  * Have enough memory
- Use IVF when:
  * Dataset is large (millions of vectors)
  * Can tolerate approximate results
  * Need faster search speed
  * Willing to spend time training once
"""





import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import faiss
import time
from mpl_toolkits.mplot3d import Axes3D

def generate_sample_vectors(n_vectors=100, n_dimensions=3, seed=42):
    """Generate synthetic vectors for demonstration."""
    np.random.seed(seed)
    vectors = np.random.randn(n_vectors, n_dimensions)
    normalized_vectors = normalize(vectors, norm='l2')
    return normalized_vectors

def plot_vectors_3d(vectors, query_vector=None, matches=None, title="Vector Space Visualization"):
    """Basic 3D visualization without regions."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all vectors
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c='blue', alpha=0.5, label='Database vectors')
    
    # Plot query vector if provided
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=100, label='Query vector')
    
    # Plot matches if provided
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_vectors_with_regions(vectors, centroids, query_vector=None, matches=None, 
                            searched_regions=None, title="Vector Space with FAISS Regions"):
    """
    Visualize vectors in 3D space with their clusters/regions.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assign each vector to nearest centroid
    distances, assignments = compute_vector_assignments(vectors, centroids)
    
    # Plot vectors colored by their cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i in range(len(centroids)):
        cluster_vectors = vectors[assignments == i]
        if len(cluster_vectors) > 0:
            # Make vectors transparent if their region wasn't searched
            alpha = 1.0 if searched_regions is None or i in searched_regions else 0.1
            ax.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], cluster_vectors[:, 2], 
                      c=[colors[i]], alpha=alpha, label=f'Region {i}')
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
              c='black', s=100, marker='*', label='Region Centers')
    
    # Plot query vector
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=200, marker='x', label='Query Vector')
    
    # Plot matches
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    plt.show()

def compute_vector_assignments(vectors, centroids):
    """Compute which vectors belong to which centroids."""
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(centroids)
    distances, assignments = index.search(vectors, 1)
    return distances, assignments.ravel()

def train_kmeans_get_centroids(vectors, n_clusters):
    """Train k-means and get centroids."""
    kmeans = faiss.Kmeans(d=vectors.shape[1], k=n_clusters, niter=20, verbose=False)
    kmeans.train(vectors)
    return kmeans.centroids



def brute_force_cosine_search(database_vectors, query_vector, k=5):
    """Perform brute force cosine similarity search."""
    start_time = time.time()
    similarities = np.dot(database_vectors, query_vector)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    end_time = time.time()
    return top_k_indices, end_time - start_time

def faiss_flat_l2_search(database_vectors, query_vector, k=5):
    """Perform basic FAISS L2 search (no regions)."""
    dimension = database_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    index.add(database_vectors)
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    end_time = time.time()
    
    return indices[0], end_time - start_time



def faiss_ivf_search_realistic(database_vectors, query_vectors, k=5, n_regions=10, nprobe=3):
    """
    More realistic FAISS IVF search that:
    - Separates training time from search time
    - Handles batch queries
    
    Consider changing function name later as this was named while experimenting with various scenarios
    """
    dimension = database_vectors.shape[1]
    
    # Create and train index (this would normally be done once and saved)
    print("Training index (this is usually done once)...")
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2)
    
    train_start = time.time()
    index.train(database_vectors)
    train_time = time.time() - train_start
    
    # Add vectors (this is also usually done once)
    add_start = time.time()
    index.add(database_vectors)
    add_time = time.time() - add_start
    
    # Set number of regions to search
    index.nprobe = nprobe
    
    # Actual search (this is what we'd do many times)
    search_start = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - search_start
    
    return indices, search_time, train_time, add_time


# 1. Generate sample data for visualization
n_vectors = 1000
n_dimensions = 3  # Using 3D for visualization
k = 5
print(f"Generating {n_vectors} vectors with {n_dimensions} dimensions...")
database_vectors = generate_sample_vectors(n_vectors, n_dimensions)

# 2. Generate a random query vector
query_vector = generate_sample_vectors(1, n_dimensions)[0]
query_vector_batch = query_vector.reshape(1, -1)  # Reshape for batch processing

# 3. Visualize initial vector space
print("\nVisualizing initial vector space...")
plot_vectors_3d(database_vectors, query_vector, title="Initial Vector Space")

# 4. Perform brute force cosine similarity search
print("\nPerforming brute force cosine similarity search...")
cosine_matches, cosine_time = brute_force_cosine_search(database_vectors, query_vector, k)
print(f"Brute force search time: {cosine_time:.6f} seconds")
print(f"Top {k} cosine similarity matches (indices): {cosine_matches}")

# 5. Visualize cosine results
plot_vectors_3d(database_vectors, query_vector, cosine_matches, 
               "Brute Force Cosine Similarity Results")

# 6. Perform basic FAISS L2 search
print("\nPerforming basic FAISS L2 search...")
faiss_matches, faiss_time = faiss_flat_l2_search(database_vectors, query_vector, k)
print(f"FAISS L2 search time: {faiss_time:.6f} seconds")
print(f"Top {k} L2 distance matches (indices): {faiss_matches}")

# 7. Visualize basic FAISS results
plot_vectors_3d(database_vectors, query_vector, faiss_matches, 
               "FAISS L2 Search Results")

# 8. Perform realistic FAISS IVF search with regions
print("\nPerforming FAISS IVF search with regions...")
n_regions = 10
nprobe = 3

# Create and train index
dimension = database_vectors.shape[1]
quantizer = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2)

print("Training index...")
train_start = time.time()
index.train(database_vectors)
train_time = time.time() - train_start
print(f"Training time: {train_time:.6f} seconds")

print("Adding vectors...")
add_start = time.time()
index.add(database_vectors)
add_time = time.time() - add_start
print(f"Adding time: {add_time:.6f} seconds")

# Set number of regions to search
index.nprobe = nprobe

# Perform search
print("Searching...")
search_start = time.time()
distances, ivf_matches = index.search(query_vector_batch, k)
search_time = time.time() - search_start
print(f"Search time: {search_time:.6f} seconds")
ivf_matches = ivf_matches[0]  # Get first batch result

# Get centroids for visualization
centroids = train_kmeans_get_centroids(database_vectors, n_regions)

# Get searched regions (approximate using nearest centroids to query)
_, searched_regions = quantizer.search(query_vector_batch, nprobe)
searched_regions = searched_regions[0]

print(f"Total time (train + add + search): {train_time + add_time + search_time:.6f} seconds")
print(f"Search-only time: {search_time:.6f} seconds")
print(f"Searched {nprobe} out of {n_regions} regions: {searched_regions}")

# 9. Visualize IVF results with regions
plot_vectors_with_regions(
    database_vectors, 
    centroids,
    query_vector, 
    ivf_matches,
    searched_regions,
    "FAISS IVF Search Results (Highlighted Searched Regions)"
)

# 10. Compare results
print("\nComparing results between methods:")
common_matches_basic = set(cosine_matches).intersection(set(faiss_matches))
common_matches_ivf = set(cosine_matches).intersection(set(ivf_matches))
print(f"Common matches (Cosine vs Basic FAISS): {len(common_matches_basic)}")
print(f"Common match indices: {common_matches_basic}")
print(f"Common matches (Cosine vs IVF FAISS): {len(common_matches_ivf)}")
print(f"Common match indices: {common_matches_ivf}")

###########################################

# Test with different vector sizes
vector_sizes = [100, 1000, 10000, 100000]
n_queries = 1000
k = 5
dimension = 128

# Dictionary to store results
results = {
    'sizes': vector_sizes,
    'brute_force': [],
    'faiss_flat': [],
    'faiss_ivf': []
}

for size in vector_sizes:
    print(f"\nTesting with {size} vectors and {n_queries} queries...")
    vectors = generate_sample_vectors(size, dimension)
    query_vectors = generate_sample_vectors(n_queries, dimension)
    
    # 1. Brute Force Cosine
    start_time = time.time()
    for query in query_vectors:
        _, _ = brute_force_cosine_search(vectors, query, k)
    brute_force_time = (time.time() - start_time) / n_queries
    results['brute_force'].append(brute_force_time)
    
    # 2. FAISS Flat L2
    start_time = time.time()
    for query in query_vectors:
        _, _ = faiss_flat_l2_search(vectors, query.reshape(1, -1), k)
    faiss_flat_time = (time.time() - start_time) / n_queries
    results['faiss_flat'].append(faiss_flat_time)
    
    # 3. FAISS IVF
    n_regions = min(size // 100, 1000)
    _, search_time, _, _ = faiss_ivf_search_realistic(vectors, query_vectors, k, n_regions)
    faiss_ivf_time = search_time / n_queries
    results['faiss_ivf'].append(faiss_ivf_time)
    
    print(f"Average times per query (seconds):")
    print(f"Brute Force: {brute_force_time:.6f}")
    print(f"FAISS Flat: {faiss_flat_time:.6f}")
    print(f"FAISS IVF: {faiss_ivf_time:.6f}")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(results['sizes'], results['brute_force'], 'o-', label='Brute Force Cosine')
plt.plot(results['sizes'], results['faiss_flat'], 's-', label='FAISS Flat L2')
plt.plot(results['sizes'], results['faiss_ivf'], '^-', label='FAISS IVF')

plt.xscale('log')  # Log scale for vector sizes
plt.yscale('log')  # Log scale for times

plt.xlabel('Number of Vectors')
plt.ylabel('Average Search Time per Query (seconds)')
plt.title('Search Time Comparison: Brute Force vs FAISS Methods')
plt.grid(True)
plt.legend()

# Add value annotations
for i, size in enumerate(results['sizes']):
    plt.annotate(f'{results["brute_force"][i]:.6f}', 
                (size, results['brute_force'][i]), 
                textcoords="offset points", xytext=(0,10))
    plt.annotate(f'{results["faiss_flat"][i]:.6f}', 
                (size, results['faiss_flat'][i]), 
                textcoords="offset points", xytext=(0,-15))
    plt.annotate(f'{results["faiss_ivf"][i]:.6f}', 
                (size, results['faiss_ivf'][i]), 
                textcoords="offset points", xytext=(0,10))

plt.tight_layout()
plt.show()
