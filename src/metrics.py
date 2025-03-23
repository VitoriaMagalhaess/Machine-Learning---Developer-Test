import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate classification accuracy.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Accuracy score.
    """
    return np.mean(y_true == y_pred)

def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """
    Calculate top-k accuracy (whether the true label is among the k most probable predictions).
    
    Args:
        y_true: True labels.
        y_proba: Predicted class probabilities.
        k: Number of top predictions to consider.
        
    Returns:
        Top-k accuracy score.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0
    
    
    top_k_indices = np.argsort(-y_proba, axis=1)[:, :k]
    
    # Check if true label is among top-k predictions
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_indices[i]:
            correct += 1
    
    return correct / n_samples

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        n_classes: Number of classes.
        
    Returns:
        Confusion matrix.
    """
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        conf_matrix[y_true[i], y_pred[i]] += 1
    
    return conf_matrix

def calculate_precision_recall_f1(confusion_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate precision, recall, and F1-score for each class.
    
    Args:
        confusion_matrix: Confusion matrix.
        
    Returns:
        Tuple containing arrays of precision, recall, and F1-scores for each class.
    """
    n_classes = confusion_matrix.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    
    for i in range(n_classes):
        # True positives
        tp = confusion_matrix[i, i]
        
        # False positives (sum of column - true positives)
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # False negatives (sum of row - true positives)
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Calculate precision
        if tp + fp == 0:
            precision[i] = 0
        else:
            precision[i] = tp / (tp + fp)
        
        # Calculate recall
        if tp + fn == 0:
            recall[i] = 0
        else:
            recall[i] = tp / (tp + fn)
        
        # Calculate F1-score
        if precision[i] + recall[i] == 0:
            f1[i] = 0
        else:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    
    return precision, recall, f1

def calculate_macro_f1(f1_scores: np.ndarray) -> float:
    """
    Calculate macro-averaged F1-score.
    
    Args:
        f1_scores: F1-scores for each class.
        
    Returns:
        Macro-averaged F1-score.
    """
    return np.mean(f1_scores)

def calculate_weighted_f1(f1_scores: np.ndarray, class_counts: np.ndarray) -> float:
    """
    Calculate weighted F1-score.
    
    Args:
        f1_scores: F1-scores for each class.
        class_counts: Number of samples per class.
        
    Returns:
        Weighted F1-score.
    """
    return np.average(f1_scores, weights=class_counts)

def calculate_roc_curve(y_true: np.ndarray, y_score: np.ndarray, positive_class: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate ROC curve for a binary classification problem.
    
    Args:
        y_true: True labels (binary).
        y_score: Predicted scores or probabilities.
        positive_class: The positive class label.
        
    Returns:
        Tuple containing false positive rates, true positive rates, and thresholds.
    """
    # Convert to binary problem
    y_true_binary = (y_true == positive_class).astype(int)
    
    # Sort by decreasing probability
    sorted_indices = np.argsort(-y_score)
    y_true_binary = y_true_binary[sorted_indices]
    
    # Calculate cumulative TP and FP
    tps = np.cumsum(y_true_binary)
    fps = np.cumsum(1 - y_true_binary)
    
    # Calculate rates
    n_pos = np.sum(y_true_binary)
    n_neg = len(y_true_binary) - n_pos
    
    if n_pos == 0:
        logger.warning(f"No positive samples for class {positive_class}")
        return np.array([0, 1]), np.array([0, 0]), np.array([1, 0])
    
    if n_neg == 0:
        logger.warning(f"No negative samples for class {positive_class}")
        return np.array([0, 0]), np.array([0, 1]), np.array([1, 0])
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0,0) and (1,1) points
    tpr = np.concatenate([[0], tpr, [1]])
    fpr = np.concatenate([[0], fpr, [1]])
    thresholds = np.concatenate([[np.inf], y_score[sorted_indices], [0]])
    
    return fpr, tpr, thresholds

def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Calculate Area Under the ROC Curve using the trapezoidal rule.
    
    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        
    Returns:
        AUC score.
    """
   
    indices = np.argsort(fpr)
    fpr_sorted = fpr[indices]
    tpr_sorted = tpr[indices]
    
    width = np.diff(fpr_sorted)
    height = (tpr_sorted[1:] + tpr_sorted[:-1]) / 2
    
    return np.sum(width * height)

def calculate_multiclass_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate ROC curves and AUC for multiclass classification using one-vs-rest approach.
    
    Args:
        y_true: True labels.
        y_proba: Predicted class probabilities.
        
    Returns:
        Dictionary containing ROC curves and AUC scores for each class.
    """
    n_classes = y_proba.shape[1]
    result = {
        'roc_curves': {},
        'auc_scores': {}
    }
   
    for i in range(n_classes):
        fpr, tpr, thresholds = calculate_roc_curve(y_true, y_proba[:, i], i)
        auc = calculate_auc(fpr, tpr)
        
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
        
        micro_auc = calculate_auc(fpr, tpr)
        result['roc_curves']['micro'] = (fpr, tpr, thresholds)
        result['auc_scores']['micro'] = micro_auc

    result['auc_scores']['macro'] = np.mean(list(result['auc_scores'].values())[:-1] if 'micro' in result['auc_scores'] else list(result['auc_scores'].values()))
    
    return result

def evaluate_cross_validation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate cross-validation results by calculating metrics for each fold.
    
    Args:
        results: Dictionary containing cross-validation results.
        
    Returns:
        Dictionary with updated metrics.
    """
    logger.info("Evaluating cross-validation results")
    
    first_key = next(iter(results['predictions']))
    first_fold = results['predictions'][first_key][0]
    y_true_first = first_fold[0]
    n_classes = len(np.unique(y_true_first))
    
    for key in results['predictions']:
        for fold, (y_true, y_pred) in enumerate(results['predictions'][key]):
           
            accuracy = calculate_accuracy(y_true, y_pred)
            results['metrics']['accuracy'][key].append(accuracy)
            
            conf_matrix = calculate_confusion_matrix(y_true, y_pred, n_classes)
            
            precision, recall, f1 = calculate_precision_recall_f1(conf_matrix)
            
            macro_f1 = calculate_macro_f1(f1)
            results['metrics']['f1_score'][key].append(macro_f1)
            
            y_true_prob, y_proba = results['probabilities'][key][fold]
            
            top_k = min(3, n_classes)
            top_k_acc = calculate_top_k_accuracy(y_true_prob, y_proba, k=top_k)
            results['metrics']['top_k_accuracy'][key].append(top_k_acc)
            
            roc_auc_results = calculate_multiclass_roc_auc(y_true_prob, y_proba)
            
            results['metrics']['auc'][key].append(roc_auc_results['auc_scores']['macro'])
    
    logger.info("Cross-validation evaluation completed")
    return results
