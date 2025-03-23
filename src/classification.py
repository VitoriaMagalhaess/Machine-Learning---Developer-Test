import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import logging
from typing import Dict, List, Tuple, Any, Callable
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KNN:
    """
    Custom implementation of K-Nearest Neighbors classifier.
    """
    
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
        """
        Initialize the KNN classifier.
        
        Args:
            k: Number of neighbors to consider.
            distance_metric: Distance metric to use ('euclidean' or 'cosine').
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        logger.info(f"Initialized KNN classifier with k={k}, distance_metric={distance_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the KNN classifier on the training data.
        
        Args:
            X: Training features.
            y: Training labels.
        """
        self.X_train = X
        self.y_train = y
        logger.debug(f"Fitted KNN classifier on data with shape X: {X.shape}, y: {y.shape}")
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            x1: First vector.
            x2: Second vector.
            
        Returns:
            Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calculate Cosine distance between two vectors.
        
        Args:
            x1: First vector.
            x2: Second vector.
            
        Returns:
            Cosine distance.
        """
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        # Avoid division by zero
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0  # Maximum distance
        
        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        # Convert to distance (1 - similarity)
        return 1.0 - cosine_similarity
    
    def _calculate_distances(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate distances between input vector and all training vectors.
        
        Args:
            x: Input vector.
            
        Returns:
            Array of distances.
        """
        distances = []
        
        for i in range(len(self.X_train)):
            if self.distance_metric == 'euclidean':
                dist = self._euclidean_distance(x, self.X_train[i])
            elif self.distance_metric == 'cosine':
                dist = self._cosine_distance(x, self.X_train[i])
            else:
                raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
            distances.append(dist)
        
        return np.array(distances)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for input features.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted labels.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        y_pred = []
        
        for x in X:
            # Calculate distances
            distances = self._calculate_distances(x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Get most common label
            most_common = np.bincount(k_nearest_labels).argmax()
            y_pred.append(most_common)
        
        return np.array(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input features.
        
        Args:
            X: Input features.
            
        Returns:
            Predicted class probabilities.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier not fitted. Call fit() first.")
        
        # Get unique classes
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        # Initialize probabilities
        probas = np.zeros((len(X), n_classes))
        
        for i, x in enumerate(X):
            # Calculate distances
            distances = self._calculate_distances(x)
            
            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            
            # Get labels of k nearest neighbors
            k_nearest_labels = self.y_train[k_indices]
            
            # Calculate class probabilities
            for j, c in enumerate(classes):
                probas[i, j] = np.mean(k_nearest_labels == c)
        
        return probas

def cross_validate(X: np.ndarray, y: np.ndarray, k_values: List[int], 
                  distance_metrics: List[str], n_folds: int = 10) -> Dict[str, Any]:
    """
    Perform cross-validation to evaluate KNN performance with different k values 
    and distance metrics.
    
    Args:
        X: Input features.
        y: Input labels.
        k_values: List of k values to evaluate.
        distance_metrics: List of distance metrics to evaluate.
        n_folds: Number of folds for cross-validation.
        
    Returns:
        Dictionary containing evaluation results.
    """
    logger.info(f"Starting {n_folds}-fold cross-validation for k values {k_values} and metrics {distance_metrics}")
    
    # Initialize results
    results = {
        'k_values': k_values,
        'distance_metrics': distance_metrics,
        'metrics': {
            'accuracy': {},
            'top_k_accuracy': {},
            'f1_score': {},
            'auc': {}
        },
        'predictions': {},
        'probabilities': {}
    }
    
    # Initialize results for each combination of k and distance metric
    for metric in distance_metrics:
        for k in k_values:
            key = f"{metric}_k{k}"
            results['metrics']['accuracy'][key] = []
            results['metrics']['top_k_accuracy'][key] = []
            results['metrics']['f1_score'][key] = []
            results['metrics']['auc'][key] = []
            results['predictions'][key] = []
            results['probabilities'][key] = []
    
    # Create stratified k-fold cross-validator
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Perform cross-validation
    fold_idx = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_idx += 1
        logger.info(f"Processing fold {fold_idx}/{n_folds}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Evaluate each combination of k and distance metric
        for metric in distance_metrics:
            for k in k_values:
                key = f"{metric}_k{k}"
                
                # Train and predict
                knn = KNN(k=k, distance_metric=metric)
                knn.fit(X_train, y_train)
                
                y_pred = knn.predict(X_test)
                y_proba = knn.predict_proba(X_test)
                
                # Store predictions and probabilities
                results['predictions'][key].append((y_test, y_pred))
                results['probabilities'][key].append((y_test, y_proba))
    
    logger.info("Cross-validation completed")
    return results

def find_optimal_k(results: Dict[str, Any]) -> Dict[str, Tuple[int, float]]:
    """
    Find the optimal k value for each distance metric based on average accuracy.
    
    Args:
        results: Dictionary containing cross-validation results.
        
    Returns:
        Dictionary mapping distance metrics to optimal k values and their accuracies.
    """
    logger.info("Finding optimal k values")
    
    optimal_k = {}
    
    for metric in results['distance_metrics']:
        # Calculate average accuracy for each k
        avg_accuracies = []
        
        for k in results['k_values']:
            key = f"{metric}_k{k}"
            avg_accuracy = np.mean(results['metrics']['accuracy'][key])
            avg_accuracies.append((k, avg_accuracy))
        
        # Find k with highest average accuracy
        optimal_k[metric] = max(avg_accuracies, key=lambda x: x[1])
        
        logger.info(f"Optimal k for {metric} distance: k={optimal_k[metric][0]} with accuracy {optimal_k[metric][1]:.4f}")
    
    return optimal_k

def compare_distance_metrics(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]]) -> Dict[str, Any]:
    """
    Compare performance between different distance metrics using optimal k values.
    
    Args:
        results: Dictionary containing cross-validation results.
        optimal_k: Dictionary mapping distance metrics to optimal k values.
        
    Returns:
        Dictionary containing comparison results.
    """
    logger.info("Comparing distance metrics")
    
    comparison = {}
    
    for metric in results['distance_metrics']:
        k = optimal_k[metric][0]
        key = f"{metric}_k{k}"
        
        comparison[metric] = {
            'accuracy': np.mean(results['metrics']['accuracy'][key]),
            'top_k_accuracy': np.mean(results['metrics']['top_k_accuracy'][key]),
            'f1_score': np.mean(results['metrics']['f1_score'][key]),
            'auc': np.mean(results['metrics']['auc'][key])
        }
    
    logger.info("Distance metrics comparison completed")
    return comparison
