import logging
import numpy as np
from typing import List, Dict, Tuple, Any
logger = logging.getLogger(__name__)
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
   
    return np.mean(y_true == y_pred)
def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    
    k = min(k, y_proba.shape[1])
    
    top_k_indices = np.argsort(-y_proba, axis=1)[:, :k]
    
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_indices[i]:
            correct += 1
    
    return correct / len(y_true)
def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
   
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    
    return confusion_matrix
def calculate_precision_recall_f1(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    n_classes = confusion_matrix.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1_scores = np.zeros(n_classes)
    
    for i in range(n_classes):

        tp = confusion_matrix[i, i]
        
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1_scores[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1_scores
def calculate_macro_f1(f1_scores: np.ndarray) -> float:
  
    return np.mean(f1_scores)
def calculate_weighted_f1(f1_scores: np.ndarray, class_counts: np.ndarray) -> float:
 
    weights = class_counts / np.sum(class_counts)
    return np.sum(f1_scores * weights)
def calculate_roc_curve(y_true: np.ndarray, y_score: np.ndarray, positive_class: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
   
    y_true_binary = (y_true == positive_class).astype(int)
    
    sorted_indices = np.argsort(-y_score)
    y_true_binary = y_true_binary[sorted_indices]
    
    tps = np.cumsum(y_true_binary)
    fps = np.cumsum(1 - y_true_binary)
    
    n_pos = np.sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos
    
    if n_pos == 0 or n_neg == 0:
       
        return np.array([0, 1]), np.array([0, 0]) if n_pos == 0 else np.array([0, 1]), np.array([1, 0])
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    thresholds = np.concatenate([[np.inf], y_score[sorted_indices], [0]])
    
    return fpr, tpr, thresholds
def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
   
    return np.trapz(tpr, fpr)
def calculate_multiclass_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
  
    n_classes = y_proba.shape[1]
    result = {
        'roc_curves': {},
        'auc_scores': {}
    }

    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_proba[:, i]
        
        sorted_indices = np.argsort(-y_score)
        y_true_binary = y_true_binary[sorted_indices]
        
        tps = np.cumsum(y_true_binary)
        fps = np.cumsum(1 - y_true_binary)
    
        n_pos = np.sum(y_true_binary)
        n_neg = len(y_true_binary) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            fpr = np.array([0, 1])
            tpr = np.array([0, 0]) if n_pos == 0 else np.array([0, 1])
            thresholds = np.array([1, 0])
        else:
            tpr = tps / n_pos
            fpr = fps / n_neg
           
            tpr = np.concatenate([[0], tpr, [1]])
            fpr = np.concatenate([[0], fpr, [1]])
            thresholds = np.concatenate([[np.inf], y_score[sorted_indices], [0]])
        
        width = np.diff(fpr)
        height = (tpr[1:] + tpr[:-1]) / 2
        auc = np.sum(width * height)
        
        result['roc_curves'][i] = (fpr, tpr, thresholds)
        result['auc_scores'][i] = auc
    
    y_true_one_hot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_one_hot[i, y_true[i]] = 1
    
    y_pred_flat = y_proba.ravel()
    y_true_flat = y_true_one_hot.ravel()
    
    sorted_indices = np.argsort(-y_pred_flat)
    y_true_flat = y_true_flat[sorted_indices]
   
    tps = np.cumsum(y_true_flat)
    fps = np.cumsum(1 - y_true_flat)
    
    n_pos = np.sum(y_true_flat)
    n_neg = len(y_true_flat) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        tpr = tps / n_pos
        fpr = fps / n_neg
        
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])
        thresholds = np.concatenate([[np.inf], y_pred_flat[sorted_indices], [0]])
        
        micro_auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        result['roc_curves']['micro'] = (fpr, tpr, thresholds)
        result['auc_scores']['micro'] = micro_auc
   
    class_aucs = [result['auc_scores'][i] for i in range(n_classes)]
    result['auc_scores']['macro'] = np.mean(class_aucs)
    
    return result
def evaluate_cross_validation_results(results: Dict[str, Any]) -> Dict[str, Any]:
   
    metrics = {
        'accuracy': {},
        'top_k_accuracy': {},
        'f1': {},
        'auc': {}
    }
    
    for key in results['predictions'].keys():
        accuracies = []
        top_k_accuracies = []
        f1_scores = []
        auc_scores = []
        
        for i in range(len(results['predictions'][key])):
            y_true = results['true_labels'][key][i]
            y_pred = results['predictions'][key][i]
            
            y_true_proba, y_proba = results['probabilities'][key][i]
        
            if not np.array_equal(y_true, y_true_proba):
                logger.warning(f"y_true e y_true_proba não são iguais para {key}, fold {i}")
           
            accuracy = calculate_accuracy(y_true, y_pred)
            accuracies.append(accuracy)
            
            top_k = min(3, y_proba.shape[1]) 
            top_k_accuracy = calculate_top_k_accuracy(y_true, y_proba, k=top_k)
            top_k_accuracies.append(top_k_accuracy)
            
            n_classes = y_proba.shape[1]
            confusion_matrix = calculate_confusion_matrix(y_true, y_pred, n_classes)
            _, _, f1 = calculate_precision_recall_f1(confusion_matrix)
            macro_f1 = calculate_macro_f1(f1)
            f1_scores.append(macro_f1)
       
            roc_auc_results = calculate_multiclass_roc_auc(y_true, y_proba)
            auc = roc_auc_results['auc_scores']['macro']
            auc_scores.append(auc)
        
        metrics['accuracy'][key] = np.mean(accuracies)
        metrics['top_k_accuracy'][key] = np.mean(top_k_accuracies)
        metrics['f1'][key] = np.mean(f1_scores)
        metrics['auc'][key] = np.mean(auc_scores)
    
    results['metrics'] = metrics
    
    return results