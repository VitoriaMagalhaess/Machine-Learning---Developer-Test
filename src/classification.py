import logging
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

logger = logging.getLogger(__name__)
class KNN:
   
    def __init__(self, k: int = 5, distance_metric: str = 'euclidean'):
       
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
        self.classes = None
        logger.info(f"Inicializado classificador KNN com k={k}, distance_metric={distance_metric}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
    
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
      
        dot_product = np.dot(x1, x2)
        norm_x1 = np.sqrt(np.sum(x1 ** 2))
        norm_x2 = np.sqrt(np.sum(x2 ** 2))
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
        
        similarity = dot_product / (norm_x1 * norm_x2)
        
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Converter similaridade para distância (1 - similaridade)
        return 1.0 - similarity
    
    def _calculate_distances(self, x: np.ndarray) -> np.ndarray:
      
        distances = []
        
        for x_train in self.X_train:
            if self.distance_metric == 'euclidean':
                distance = self._euclidean_distance(x, x_train)
            elif self.distance_metric == 'cosine':
                distance = self._cosine_distance(x, x_train)
            else:
                raise ValueError(f"Métrica de distância não suportada: {self.distance_metric}")
            
            distances.append(distance)
        
        return np.array(distances)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        
        y_pred = []
        
        for x in X:
            distances = self._calculate_distances(x)
            
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            most_common_index = np.argmax(counts)
            most_common_label = unique_labels[most_common_index]
            
            y_pred.append(most_common_label)
        
        return np.array(y_pred)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
      
        probabilities = []
        
        for x in X:
            
            distances = self._calculate_distances(x)
           
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            class_counts = np.zeros(len(self.classes))

            for i, class_label in enumerate(self.classes):
                class_counts[i] = np.sum(k_nearest_labels == class_label)
          
            probabilities.append(class_counts / self.k)
        
        return np.array(probabilities)
def cross_validate(X: np.ndarray, y: np.ndarray, k_values: List[int], 
                  distance_metrics: List[str], n_folds: int = 10) -> Dict[str, Any]:
  
    results = {
        'predictions': {},
        'probabilities': {},
        'true_labels': {},
        'metrics': {
            'accuracy': {},
            'top_k_accuracy': {},
            'f1': {},
            'auc': {}
        }
    }
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for metric in distance_metrics:
        for k in k_values:
            key = f"{metric}_k{k}"
            results['predictions'][key] = []
            results['probabilities'][key] = []
            results['true_labels'][key] = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Processando fold {fold+1}/{n_folds}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        for metric in distance_metrics:
            for k in k_values:

                knn = KNN(k=k, distance_metric=metric)
                knn.fit(X_train, y_train)
            
                y_pred = knn.predict(X_test)
                
                y_proba = knn.predict_proba(X_test)
                
                key = f"{metric}_k{k}"
                results['predictions'][key].append(y_pred)
                results['probabilities'][key].append((y_test, y_proba))
                results['true_labels'][key].append(y_test)
    
    logger.info("Validação cruzada concluída")
    return results
def find_optimal_k(results: Dict[str, Any]) -> Dict[str, Tuple[int, float]]:
    
    optimal_k = {}
    
    for key in results['predictions'].keys():
     
        parts = key.split('_')
        method = parts[0]
        k = int(parts[1][1:])  # Remover 'k' do início
        
    
        accuracies = []
        for i in range(len(results['predictions'][key])):
            y_true = results['true_labels'][key][i]
            y_pred = results['predictions'][key][i]
           
            accuracy = np.mean(y_true == y_pred)
            accuracies.append(accuracy)
        
        mean_accuracy = np.mean(accuracies)
        
        if method not in optimal_k or optimal_k[method][1] < mean_accuracy:
            optimal_k[method] = (k, mean_accuracy)
    
    return optimal_k
def compare_distance_metrics(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]]) -> Dict[str, Any]:
   
    comparison = {}
    
    for metric, (k, _) in optimal_k.items():
        key = f"{metric}_k{k}"
        
        accuracy = results['metrics']['accuracy'][key]
        top_k_accuracy = results['metrics']['top_k_accuracy'][key]
        f1 = results['metrics']['f1'][key]
        auc = results['metrics']['auc'][key]
        
        comparison[metric] = {
            'accuracy': accuracy,
            'top_k_accuracy': top_k_accuracy,
            'f1': f1,
            'auc': auc
        }
    
    return comparison