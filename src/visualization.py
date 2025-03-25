import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from typing import Dict, Tuple, List, Any, Optional

logger = logging.getLogger(__name__)

def apply_tsne(embeddings: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:

    if perplexity >= embeddings.shape[0]:
        perplexity = max(5, embeddings.shape[0] // 5)
        logger.warning(f"Perplexidade ajustada para {perplexity} devido ao pequeno conjunto de dados")
    
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
        init='pca' 
    )
    
    try:
        reduced_embeddings = tsne.fit_transform(embeddings)
        return reduced_embeddings
    except Exception as e:
        logger.error(f"Erro ao aplicar t-SNE: {e}")
       
        logger.info("Tentando novamente com parâmetros mais conservadores")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, 5),
            n_iter=250,
            random_state=42,
            method='exact',  
            init='random'
        )
        reduced_embeddings = tsne.fit_transform(embeddings)
        return reduced_embeddings

def visualize_embeddings(reduced_embeddings: np.ndarray, 
                        labels: np.ndarray, 
                        label_mapping: Dict[int, Any] = None,
                        title: str = 't-SNE Visualization of Embeddings by Syndrome ID',
                        save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
  
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(12, 10), dpi=150)
    
    unique_labels = np.unique(labels)
    
    cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    
    alpha = 0.7
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        points = reduced_embeddings[mask]
        
        if label_mapping is not None:
            if hasattr(label, 'dtype') and np.issubdtype(label.dtype, np.integer):
                label_key = int(label)
            else:
                try:
                    label_key = int(label)
                except (TypeError, ValueError):
                    label_key = label
                    
            legend_label = label_mapping[label_key]
        else:
            legend_label = f"Class {label}"
        
        plt.scatter(points[:, 0], points[:, 1], 
                   alpha=alpha, 
                   s=50, 
                   c=[cmap(i)], 
                   label=legend_label, 
                   edgecolors='w', 
                   linewidths=0.5)
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='best', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualização salva em {save_path}")
    
    return plt.gcf(), plt.gca()

def analyze_clustering(reduced_embeddings: np.ndarray, 
                      labels: np.ndarray, 
                      label_mapping: Dict[int, Any] = None) -> Dict[str, Any]:
    
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)
    
    cluster_centers = {}
    cluster_dispersions = {}
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = reduced_embeddings[mask]
        
        center = np.mean(cluster_points, axis=0)
        
        distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
        dispersion = np.mean(distances)
        
        if hasattr(label, 'dtype') and np.issubdtype(label.dtype, np.integer):
            label_key = int(label)
        else:
            try:
                label_key = int(label)
            except (TypeError, ValueError):
                label_key = label
                
        cluster_name = label_mapping[label_key] if label_mapping else f"Cluster {label}"
        cluster_centers[cluster_name] = center
        cluster_dispersions[cluster_name] = dispersion
    
    center_points = np.array(list(cluster_centers.values()))
    cluster_names = list(cluster_centers.keys())
    
    distance_matrix = squareform(pdist(center_points, 'euclidean'))
    
    num_clusters = len(cluster_names)
    
    closest_distance = float('inf')
    closest_pair = None
    furthest_distance = 0
    furthest_pair = None
    
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            distance = distance_matrix[i, j]
            
            if distance < closest_distance:
                closest_distance = distance
                closest_pair = (cluster_names[i], cluster_names[j])
            
            if distance > furthest_distance:
                furthest_distance = distance
                furthest_pair = (cluster_names[i], cluster_names[j])
    
    mean_dispersion = np.mean(list(cluster_dispersions.values()))
    overlap_threshold = 2 * mean_dispersion  
    
    potential_overlaps = []
    for i in range(num_clusters):
        for j in range(i+1, num_clusters):
            distance = distance_matrix[i, j]
            combined_dispersion = cluster_dispersions[cluster_names[i]] + cluster_dispersions[cluster_names[j]]
            
            if distance < combined_dispersion:
                potential_overlaps.append((cluster_names[i], cluster_names[j], distance))
    
    cluster_densities = {}
    for label in unique_labels:
        mask = labels == label
        cluster_points = reduced_embeddings[mask]
        
        if hasattr(label, 'dtype') and np.issubdtype(label.dtype, np.integer):
            label_key = int(label)
        else:
            try:
                label_key = int(label)
            except (TypeError, ValueError):
                label_key = label
                
        cluster_name = label_mapping[label_key] if label_mapping else f"Cluster {label}"
        if cluster_dispersions[cluster_name] > 0:
            density = len(cluster_points) / (cluster_dispersions[cluster_name] ** 2 * np.pi)
        else:
            density = float('inf') 
        
        cluster_densities[cluster_name] = density
   
    analysis_results = {
        'number_of_clusters': num_clusters,
        'cluster_centers': cluster_centers,
        'cluster_dispersions': cluster_dispersions,
        'cluster_densities': cluster_densities,
        'closest_clusters': (closest_pair, closest_distance) if closest_pair else None,
        'furthest_clusters': (furthest_pair, furthest_distance) if furthest_pair else None,
        'potential_overlaps': potential_overlaps,
        'potential_overlap_count': len(potential_overlaps)
    }
    
    return analysis_results