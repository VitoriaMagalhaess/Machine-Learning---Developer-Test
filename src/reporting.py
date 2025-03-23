import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dataset_statistics_report(stats: Dict[str, Any]) -> str:
    """
    Generate a report of dataset statistics.
    
    Args:
        stats: Dictionary containing dataset statistics.
        
    Returns:
        String containing the report.
    """
    logger.info("Generating dataset statistics report")
    
    report = "# Dataset Statistics Report\n\n"
    
    report += f"## General Statistics\n"
    report += f"- Total number of images: {stats['total_images']}\n"
    report += f"- Number of syndromes: {stats['number_of_syndromes']}\n"
    report += f"- Number of subjects: {stats['number_of_subjects']}\n"
    report += f"- Embedding dimension: {stats['embedding_dimension']}\n\n"
    
    report += f"## Distribution Statistics\n"
    report += f"### Images per Syndrome\n"
    report += f"- Mean: {stats['images_per_syndrome']['mean']:.2f}\n"
    report += f"- Min: {stats['images_per_syndrome']['min']:.0f}\n"
    report += f"- Max: {stats['images_per_syndrome']['max']:.0f}\n"
    report += f"- Std: {stats['images_per_syndrome']['std']:.2f}\n\n"
    
    report += f"### Subjects per Syndrome\n"
    report += f"- Mean: {stats['subjects_per_syndrome']['mean']:.2f}\n"
    report += f"- Min: {stats['subjects_per_syndrome']['min']:.0f}\n"
    report += f"- Max: {stats['subjects_per_syndrome']['max']:.0f}\n"
    report += f"- Std: {stats['subjects_per_syndrome']['std']:.2f}\n\n"
    
    # Create a table of syndrome distribution
    report += f"## Syndrome Distribution\n"
    report += "| Syndrome ID | Number of Images |\n"
    report += "| ----------- | ---------------- |\n"
    
    for syndrome_id, count in sorted(stats['syndrome_distribution'].items(), key=lambda x: x[1], reverse=True):
        report += f"| {syndrome_id} | {count} |\n"
    
    logger.info("Dataset statistics report generated")
    return report

def generate_clustering_analysis_report(analysis: Dict[str, Any]) -> str:
    """
    Generate a report of clustering analysis.
    
    Args:
        analysis: Dictionary containing clustering analysis results.
        
    Returns:
        String containing the report.
    """
    logger.info("Generating clustering analysis report")
    
    report = "# Clustering Analysis Report\n\n"
    
    report += f"## General Statistics\n"
    report += f"- Number of clusters: {analysis['number_of_clusters']}\n"
    report += f"- Potential cluster overlaps: {analysis['potential_overlap_count']}\n\n"
    
    report += f"## Cluster Relationships\n"
    report += f"### Closest Clusters\n"
    report += f"- Pair: {analysis['closest_clusters']['pair']}\n"
    report += f"- Distance: {analysis['closest_clusters']['distance']:.4f}\n\n"
    
    report += f"### Furthest Clusters\n"
    report += f"- Pair: {analysis['furthest_clusters']['pair']}\n"
    report += f"- Distance: {analysis['furthest_clusters']['distance']:.4f}\n\n"
    
    report += f"## Cluster Dispersions\n"
    report += "| Cluster (Syndrome ID) | Dispersion |\n"
    report += "| --------------------- | ---------- |\n"
    
    for cluster, dispersion in analysis['cluster_dispersions']:
        report += f"| {cluster} | {dispersion:.4f} |\n"
    
    logger.info("Clustering analysis report generated")
    return report

def generate_cross_validation_report(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]]) -> str:
    """
    Generate a report of cross-validation results.
    
    Args:
        results: Dictionary containing cross-validation results.
        optimal_k: Dictionary containing optimal k values for each distance metric.
        
    Returns:
        String containing the report.
    """
    logger.info("Generating cross-validation report")
    
    report = "# Cross-Validation Results Report\n\n"
    
    report += f"## Optimal k Values\n"
    report += "| Distance Metric | Optimal k | Accuracy |\n"
    report += "| --------------- | --------- | -------- |\n"
    
    for metric, (k, accuracy) in optimal_k.items():
        report += f"| {metric} | {k} | {accuracy:.4f} |\n"
    
    report += "\n## Performance Metrics for Each Configuration\n"
    
    # Group results by distance metric
    for metric in results['distance_metrics']:
        report += f"\n### {metric.capitalize()} Distance Metric\n"
        report += "| k | Accuracy | Top-k Accuracy | F1-Score | AUC |\n"
        report += "| - | -------- | -------------- | -------- | --- |\n"
        
        for k in results['k_values']:
            key = f"{metric}_k{k}"
            
            # Calculate averages across folds
            avg_accuracy = np.mean(results['metrics']['accuracy'][key])
            avg_top_k_accuracy = np.mean(results['metrics']['top_k_accuracy'][key])
            avg_f1_score = np.mean(results['metrics']['f1_score'][key])
            avg_auc = np.mean(results['metrics']['auc'][key])
            
            report += f"| {k} | {avg_accuracy:.4f} | {avg_top_k_accuracy:.4f} | {avg_f1_score:.4f} | {avg_auc:.4f} |\n"
    
    logger.info("Cross-validation report generated")
    return report

def generate_comparison_report(comparison: Dict[str, Dict[str, float]]) -> str:
    """
    Generate a report comparing different distance metrics.
    
    Args:
        comparison: Dictionary containing comparison results.
        
    Returns:
        String containing the report.
    """
    logger.info("Generating distance metrics comparison report")
    
    report = "# Distance Metrics Comparison Report\n\n"
    
    report += "| Distance Metric | Accuracy | Top-k Accuracy | F1-Score | AUC |\n"
    report += "| --------------- | -------- | -------------- | -------- | --- |\n"
    
    for metric, metrics in comparison.items():
        report += f"| {metric.capitalize()} | {metrics['accuracy']:.4f} | {metrics['top_k_accuracy']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc']:.4f} |\n"
    
    # Determine the better metric
    metrics_list = list(comparison.keys())
    if len(metrics_list) >= 2:
        metric1, metric2 = metrics_list[0], metrics_list[1]
        
        report += "\n## Comparison Analysis\n"
        
        # Compare accuracy
        if comparison[metric1]['accuracy'] > comparison[metric2]['accuracy']:
            better_metric_acc = metric1
            diff_acc = comparison[metric1]['accuracy'] - comparison[metric2]['accuracy']
        else:
            better_metric_acc = metric2
            diff_acc = comparison[metric2]['accuracy'] - comparison[metric1]['accuracy']
        
        report += f"- **Accuracy**: {better_metric_acc.capitalize()} distance performs better by {diff_acc:.4f}\n"
        
        # Compare AUC
        if comparison[metric1]['auc'] > comparison[metric2]['auc']:
            better_metric_auc = metric1
            diff_auc = comparison[metric1]['auc'] - comparison[metric2]['auc']
        else:
            better_metric_auc = metric2
            diff_auc = comparison[metric2]['auc'] - comparison[metric1]['auc']
        
        report += f"- **AUC**: {better_metric_auc.capitalize()} distance performs better by {diff_auc:.4f}\n"
        
        # Compare F1-Score
        if comparison[metric1]['f1_score'] > comparison[metric2]['f1_score']:
            better_metric_f1 = metric1
            diff_f1 = comparison[metric1]['f1_score'] - comparison[metric2]['f1_score']
        else:
            better_metric_f1 = metric2
            diff_f1 = comparison[metric2]['f1_score'] - comparison[metric1]['f1_score']
        
        report += f"- **F1-Score**: {better_metric_f1.capitalize()} distance performs better by {diff_f1:.4f}\n"
        
        # Overall recommendation
        metrics_count = {metric1: 0, metric2: 0}
        metrics_count[better_metric_acc] += 1
        metrics_count[better_metric_auc] += 1
        metrics_count[better_metric_f1] += 1
        
        if metrics_count[metric1] > metrics_count[metric2]:
            recommended_metric = metric1
        else:
            recommended_metric = metric2
        
        report += f"\n### Recommendation\n"
        report += f"Based on the evaluation metrics, the **{recommended_metric.capitalize()} distance** metric appears to perform better overall for this classification task.\n"
    
    logger.info("Distance metrics comparison report generated")
    return report

    logger.info("Plotting ROC curves")
    
    plt.figure(figsize=(10, 8))
    
    # Colors for different distance metrics
    colors = {
        'euclidean': 'blue',
        'cosine': 'red'
    }
    
    # Get number of classes
    first_key = next(iter(results['probabilities']))
    first_fold = results['probabilities'][first_key][0]
    y_true_first, y_proba_first = first_fold
    n_classes = y_proba_first.shape[1]
    
    # Plot ROC curve for each distance metric using optimal k
    for metric, (k, _) in optimal_k.items():
        key = f"{metric}_k{k}"
        
        # Calculate average ROC curve across folds
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        # Process each fold
        for fold, (y_true, y_proba) in enumerate(results['probabilities'][key]):
            # Calculate micro-average ROC curve
            roc_auc_results = calculate_multiclass_roc_auc(y_true, y_proba)
            fpr, tpr, _ = roc_auc_results['roc_curves']['micro']
            
            # Interpolate
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        
        # Average the interpolated curves
        mean_tpr /= len(results['probabilities'][key])
        
        # Ensure the curve begins at (0, 0) and ends at (1, 1)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        
        # Calculate AUC for the averaged curve
        mean_auc = calculate_auc(mean_fpr, mean_tpr)
        
        # Plot the average ROC curve
        plt.plot(
            mean_fpr, mean_tpr,
            color=colors[metric],
            label=f'{metric.capitalize()} (k={k}, AUC={mean_auc:.4f})',
            lw=2, alpha=0.8
        )
    
    # Plot the random guessing line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    
    # Customize the plot
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add AUC score as text
    for i, (metric, (k, _)) in enumerate(optimal_k.items()):
        key = f"{metric}_k{k}"
        avg_auc = np.mean(results['metrics']['auc'][key])
        plt.text(0.05, 0.05 + 0.05 * i, f'{metric.capitalize()} AUC: {avg_auc:.4f}',
                transform=plt.gca().transAxes, color=colors[metric], fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves plot saved to {save_path}")
    
    logger.info("ROC curves plotting completed")
    return plt.gcf()

def create_summary_report(stats: Dict[str, Any], analysis: Dict[str, Any], 
                         results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]], 
                         comparison: Dict[str, Dict[str, float]]) -> str:
    """
    Create a comprehensive summary report combining all analyses.
    
    Args:
        stats: Dictionary containing dataset statistics.
        analysis: Dictionary containing clustering analysis results.
        results: Dictionary containing cross-validation results.
        optimal_k: Dictionary containing optimal k values for each distance metric.
        comparison: Dictionary containing comparison results.
        
    Returns:
        String containing the summary report.
    """
    logger.info("Creating comprehensive summary report")
    
    report = "# Genetic Syndrome Classification Report\n\n"
    
    report += "## Executive Summary\n\n"
    report += "This report presents the analysis and classification of genetic syndrome embeddings derived from images. "
    report += "The embeddings are 320-dimensional vectors representing genetic syndrome characteristics. "
    report += "We performed data preprocessing, visualization using t-SNE, and classification using KNN with both Euclidean and Cosine distance metrics.\n\n"
    
    # Add key findings
    report += "### Key Findings\n\n"
    
    # Dataset statistics
    report += f"- The dataset contains {stats['total_images']} images from {stats['number_of_syndromes']} different genetic syndromes.\n"
    report += f"- Each syndrome has an average of {stats['images_per_syndrome']['mean']:.2f} images from {stats['subjects_per_syndrome']['mean']:.2f} subjects.\n"
    
    # Add most imbalanced syndromes
    max_syndrome = max(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    min_syndrome = min(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    report += f"- The dataset shows some imbalance, with syndrome {max_syndrome[0]} having {max_syndrome[1]} images and syndrome {min_syndrome[0]} having only {min_syndrome[1]} images.\n"
    
    # t-SNE visualization findings
    report += f"- t-SNE visualization revealed {analysis['number_of_clusters']} distinct clusters with {analysis['potential_overlap_count']} potential overlaps.\n"
    
    # Classification performance
    best_metric = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    best_metric_name = best_metric[0]
    best_k = optimal_k[best_metric_name][0]
    best_accuracy = best_metric[1]['accuracy']
    best_auc = best_metric[1]['auc']
    
    report += f"- The {best_metric_name.capitalize()} distance metric with k={best_k} performed best, achieving an accuracy of {best_accuracy:.4f} and AUC of {best_auc:.4f}.\n\n"
    
    # Include the individual reports
    report += "## Detailed Analysis\n\n"
    
    # Dataset statistics report
    report += "### 1. Dataset Statistics\n\n"
    dataset_report = generate_dataset_statistics_report(stats)
    report += "\n".join(dataset_report.split("\n")[1:])  # Skip the title
    
    # Clustering analysis report
    report += "\n\n### 2. Clustering Analysis\n\n"
    clustering_report = generate_clustering_analysis_report(analysis)
    report += "\n".join(clustering_report.split("\n")[1:])  # Skip the title
    
    # Cross-validation report
    report += "\n\n### 3. Cross-Validation Results\n\n"
    cv_report = generate_cross_validation_report(results, optimal_k)
    report += "\n".join(cv_report.split("\n")[1:])  # Skip the title
    
    # Distance metrics comparison report
    report += "\n\n### 4. Distance Metrics Comparison\n\n"
    comparison_report = generate_comparison_report(comparison)
    report += "\n".join(comparison_report.split("\n")[1:])  # Skip the title
    
    # Conclusion
    report += "\n\n## Conclusion and Recommendations\n\n"
    report += "Based on the comprehensive analysis performed in this study, we can draw the following conclusions:\n\n"
    
    # Summarize key findings
    report += f"1. The {best_metric_name.capitalize()} distance metric outperformed the other metrics for KNN classification of genetic syndromes.\n"
    report += f"2. The optimal value of k for the KNN classifier was determined to be {best_k} through cross-validation.\n"
    report += "3. t-SNE visualization showed that some syndromes form distinct clusters, while others have potential overlap, which may affect classification performance.\n\n"
    
    # Recommendations
    report += "### Recommendations\n\n"
    report += "1. **Distance Metric Selection**: Use the " + best_metric_name.capitalize() + " distance metric for KNN classification of genetic syndrome embeddings.\n"
    report += f"2. **Parameter Tuning**: Set k={best_k} for optimal performance in the KNN classifier.\n"
    
    # Address data imbalance if present
    if stats['images_per_syndrome']['std'] > 0.5 * stats['images_per_syndrome']['mean']:
        report += "3. **Data Imbalance**: Address the imbalance in the dataset by collecting more samples for underrepresented syndromes or using techniques like SMOTE for synthetic sample generation.\n"
    
    # Suggest further improvements
    report += f"4. **Further Improvements**: Consider exploring other classification algorithms like SVM or ensemble methods to potentially improve performance beyond the achieved accuracy of {best_accuracy:.4f}.\n"
    report += "5. **Feature Engineering**: Investigate whether dimensionality reduction or feature selection could improve classification by reducing noise in the embeddings.\n"
    
    logger.info("Comprehensive summary report created")
    return report

def calculate_multiclass_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculate ROC curves and AUC for multiclass classification using one-vs-rest approach.
    This is a duplicate of the function in metrics.py to avoid circular imports.
    
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
    
    # Calculate binary ROC curve for each class
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        y_score = y_proba[:, i]
        
        # Sort by decreasing probability
        sorted_indices = np.argsort(-y_score)
        y_true_binary = y_true_binary[sorted_indices]
        
        # Calculate cumulative TP and FP
        tps = np.cumsum(y_true_binary)
        fps = np.cumsum(1 - y_true_binary)
        
        # Calculate rates
        n_pos = np.sum(y_true_binary)
        n_neg = len(y_true_binary) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            fpr = np.array([0, 1])
            tpr = np.array([0, 0]) if n_pos == 0 else np.array([0, 1])
            thresholds = np.array([1, 0])
        else:
            tpr = tps / n_pos
            fpr = fps / n_neg
            
            # Add (0,0) and (1,1) points
            tpr = np.concatenate([[0], tpr, [1]])
            fpr = np.concatenate([[0], fpr, [1]])
            thresholds = np.concatenate([[np.inf], y_score[sorted_indices], [0]])
        
        # Calculate AUC using the trapezoidal rule
        width = np.diff(fpr)
        height = (tpr[1:] + tpr[:-1]) / 2
        auc = np.sum(width * height)
        
        result['roc_curves'][i] = (fpr, tpr, thresholds)
        result['auc_scores'][i] = auc
    
    # Calculate micro-average
    y_true_one_hot = np.zeros((len(y_true), n_classes))
    for i in range(len(y_true)):
        y_true_one_hot[i, y_true[i]] = 1
    
    y_pred_flat = y_proba.ravel()
    y_true_flat = y_true_one_hot.ravel()
    
    # Sort by decreasing probability
    sorted_indices = np.argsort(-y_pred_flat)
    y_true_flat = y_true_flat[sorted_indices]
    
    # Calculate cumulative TP and FP
    tps = np.cumsum(y_true_flat)
    fps = np.cumsum(1 - y_true_flat)
    
    # Calculate rates
    n_pos = np.sum(y_true_flat)
    n_neg = len(y_true_flat) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        tpr = tps / n_pos
        fpr = fps / n_neg
        
        # Add (0,0) and (1,1) points
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])
        thresholds = np.concatenate([[np.inf], y_pred_flat[sorted_indices], [0]])
        
        micro_auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
        result['roc_curves']['micro'] = (fpr, tpr, thresholds)
        result['auc_scores']['micro'] = micro_auc
    
    # Calculate macro-average AUC
    class_aucs = [result['auc_scores'][i] for i in range(n_classes)]
    result['auc_scores']['macro'] = np.mean(class_aucs)
    
    return result
