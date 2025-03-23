import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_tsne(embeddings: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
    """
    Apply t-SNE to reduce dimensionality of embeddings to 2D.
    
    Args:
        embeddings: Array of embeddings.
        perplexity: Perplexity parameter for t-SNE.
        n_iter: Number of iterations for t-SNE.
        
    Returns:
        2D array of reduced embeddings.
    """
    logger.info(f"Applying t-SNE with perplexity={perplexity}, n_iter={n_iter}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    logger.info(f"t-SNE completed. Output shape: {reduced_embeddings.shape}")
    return reduced_embeddings

def visualize_embeddings(reduced_embeddings: np.ndarray, 
                        labels: np.ndarray, 
                        label_mapping: Dict[int, Any] = None,
                        title: str = 't-SNE Visualization of Embeddings by Syndrome ID',
                        save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate a plot that visualizes the embeddings colored by their syndrome_id.
    
    Args:
        reduced_embeddings: 2D array of reduced embeddings.
        labels: Array of labels (syndrome_ids).
        label_mapping: Dictionary mapping from integer labels to original syndrome_ids.
        title: Title of the plot.
        save_path: Path to save the figure. If None, the figure is not saved.
        
    Returns:
        Figure and Axes objects for further customization if needed.
    """
    logger.info("Generating t-SNE visualization")
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    # Create a figure
    plt.figure(figsize=(12, 10))
    
    # Plot each syndrome with a different color
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            c=[colors[i]],
            label=label_mapping[label] if label_mapping else label,
            alpha=0.7,
            s=50
        )
    
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # Add legend with smaller font size if there are many labels
    if len(unique_labels) > 10:
        plt.legend(fontsize=8, markerscale=0.7, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(fontsize=10, markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {save_path}")
    
    logger.info("t-SNE visualization completed")
    return plt.gcf(), plt.gca()

def analyze_clustering(reduced_embeddings: np.ndarray, 
                      labels: np.ndarray, 
                      label_mapping: Dict[int, Any] = None) -> Dict[str, Any]:
    """
    Analyze the clustering of embeddings in the t-SNE visualization.
    
    Args:
        reduced_embeddings: 2D array of reduced embeddings.
        labels: Array of labels (syndrome_ids).
        label_mapping: Dictionary mapping from integer labels to original syndrome_ids.
        
    Returns:
        Dictionary containing analysis results.
    """
    logger.info("Analyzing embedding clusters")
    
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    
    # Calculate centroid for each cluster
    centroids = []
    for label in unique_labels:
        mask = labels == label
        centroid = np.mean(reduced_embeddings[mask], axis=0)
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    
    # Calculate distances between centroids
    centroid_distances = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            centroid_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    # Find closest and furthest pairs
    i, j = np.unravel_index(np.argmin(centroid_distances + np.eye(n_labels) * 1e10), centroid_distances.shape)
    closest_pair = (label_mapping[unique_labels[i]] if label_mapping else unique_labels[i],
                   label_mapping[unique_labels[j]] if label_mapping else unique_labels[j])
    closest_distance = centroid_distances[i, j]
    
    i, j = np.unravel_index(np.argmax(centroid_distances), centroid_distances.shape)
    furthest_pair = (label_mapping[unique_labels[i]] if label_mapping else unique_labels[i],
                    label_mapping[unique_labels[j]] if label_mapping else unique_labels[j])
    furthest_distance = centroid_distances[i, j]
    
    # Calculate cluster dispersion (average distance from points to centroid)
    dispersions = []
    for i, label in enumerate(unique_labels):
        mask = labels == label
        points = reduced_embeddings[mask]
        dists = np.linalg.norm(points - centroids[i], axis=1)
        dispersion = np.mean(dists)
        dispersions.append((label_mapping[label] if label_mapping else label, dispersion))
    
    # Sort dispersions from highest to lowest
    dispersions.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate overlap between clusters (approximate)
    overlap_count = 0
    for i, label_i in enumerate(unique_labels):
        mask_i = labels == label_i
        points_i = reduced_embeddings[mask_i]
        radius_i = np.max(np.linalg.norm(points_i - centroids[i], axis=1))
        
        for j, label_j in enumerate(unique_labels):
            if i >= j:  # Skip self-comparisons and duplicates
                continue
            
            mask_j = labels == label_j
            points_j = reduced_embeddings[mask_j]
            radius_j = np.max(np.linalg.norm(points_j - centroids[j], axis=1))
            
            # Check if clusters potentially overlap
            centroid_dist = np.linalg.norm(centroids[i] - centroids[j])
            if centroid_dist < (radius_i + radius_j):
                overlap_count += 1
    
    analysis = {
        'number_of_clusters': n_labels,
        'closest_clusters': {
            'pair': closest_pair,
            'distance': closest_distance
        },
        'furthest_clusters': {
            'pair': furthest_pair,
            'distance': furthest_distance
        },
        'cluster_dispersions': dispersions,
        'potential_overlap_count': overlap_count
    }
    
    logger.info(f"Cluster analysis completed. Found {n_labels} clusters with {overlap_count} potential overlaps.")
    return analysis
