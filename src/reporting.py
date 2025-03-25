import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_dataset_statistics_report(stats: Dict[str, Any]) -> str:
    
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
    
    report += f"## Syndrome Distribution\n"
    report += "| Syndrome ID | Number of Images |\n"
    report += "| ----------- | ---------------- |\n"
    
    for syndrome_id, count in sorted(stats['syndrome_distribution'].items(), key=lambda x: x[1], reverse=True):
        report += f"| {syndrome_id} | {count} |\n"
    
    logger.info("Dataset statistics report generated")
    return report

def generate_clustering_analysis_report(analysis: Dict[str, Any]) -> str:
    
    logger.info("Generating clustering analysis report")
    
    report = "# Clustering Analysis Report\n\n"
    
    report += f"## General Statistics\n"
    report += f"- Number of clusters: {analysis['number_of_clusters']}\n"
    report += f"- Potential cluster overlaps: {analysis['potential_overlap_count']}\n\n"
    
    report += f"## Cluster Relationships\n"
    report += f"### Closest Clusters\n"
    
    if analysis['closest_clusters']:
        closest_pair, closest_distance = analysis['closest_clusters']
        report += f"- Pair: {closest_pair}\n"
        report += f"- Distance: {closest_distance:.4f}\n\n"
    else:
        report += "No closest clusters found.\n\n"
    
    report += f"### Furthest Clusters\n"
    
    if analysis['furthest_clusters']:
        furthest_pair, furthest_distance = analysis['furthest_clusters']
        report += f"- Pair: {furthest_pair}\n"
        report += f"- Distance: {furthest_distance:.4f}\n\n"
    else:
        report += "No furthest clusters found.\n\n"
    
    report += f"## Cluster Dispersions\n"
    report += "| Cluster (Syndrome ID) | Dispersion |\n"
    report += "| --------------------- | ---------- |\n"
    
    for cluster, dispersion in analysis['cluster_dispersions'].items():
        report += f"| {cluster} | {dispersion:.4f} |\n"
    
    logger.info("Clustering analysis report generated")
    return report

def generate_cross_validation_report(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]]) -> str:
    
    logger.info("Generating cross-validation report")
    
    report = "# Cross-Validation Results Report\n\n"
    
    report += f"## Optimal k Values\n"
    report += "| Distance Metric | Optimal k | Accuracy |\n"
    report += "| --------------- | --------- | -------- |\n"
    
    for metric, (k, accuracy) in optimal_k.items():
        report += f"| {metric} | {k} | {accuracy:.4f} |\n"
    
    report += "\n## Performance Metrics for Each Configuration\n"
    
    for metric in results['distance_metrics']:
        report += f"\n### {metric.capitalize()} Distance Metric\n"
        report += "| k | Accuracy | Top-k Accuracy | F1-Score | AUC |\n"
        report += "| - | -------- | -------------- | -------- | --- |\n"
        
        for k in results['k_values']:
            key = f"{metric}_k{k}"
            
            avg_accuracy = np.mean(results['metrics']['accuracy'][key])
            avg_top_k_accuracy = np.mean(results['metrics']['top_k_accuracy'][key])
            avg_f1_score = np.mean(results['metrics']['f1_score'][key])
            avg_auc = np.mean(results['metrics']['auc'][key])
            
            report += f"| {k} | {avg_accuracy:.4f} | {avg_top_k_accuracy:.4f} | {avg_f1_score:.4f} | {avg_auc:.4f} |\n"
    
    logger.info("Cross-validation report generated")
    return report

def generate_comparison_report(comparison: Dict[str, Dict[str, float]]) -> str:
   
    logger.info("Generating distance metrics comparison report")
    
    report = "# Distance Metrics Comparison Report\n\n"
    
    report += "| Distance Metric | Accuracy | Top-k Accuracy | F1-Score | AUC |\n"
    report += "| --------------- | -------- | -------------- | -------- | --- |\n"
    
    for metric, metrics in comparison.items():
        report += f"| {metric.capitalize()} | {metrics['accuracy']:.4f} | {metrics['top_k_accuracy']:.4f} | {metrics['f1_score']:.4f} | {metrics['auc']:.4f} |\n"
    
    metrics_list = list(comparison.keys())
    if len(metrics_list) >= 2:
        metric1, metric2 = metrics_list[0], metrics_list[1]
        
        report += "\n## Comparison Analysis\n"
        
        if comparison[metric1]['accuracy'] > comparison[metric2]['accuracy']:
            better_metric_acc = metric1
            diff_acc = comparison[metric1]['accuracy'] - comparison[metric2]['accuracy']
        else:
            better_metric_acc = metric2
            diff_acc = comparison[metric2]['accuracy'] - comparison[metric1]['accuracy']
        
        report += f"- **Accuracy**: {better_metric_acc.capitalize()} distance performs better by {diff_acc:.4f}\n"
        
        if comparison[metric1]['auc'] > comparison[metric2]['auc']:
            better_metric_auc = metric1
            diff_auc = comparison[metric1]['auc'] - comparison[metric2]['auc']
        else:
            better_metric_auc = metric2
            diff_auc = comparison[metric2]['auc'] - comparison[metric1]['auc']
        
        report += f"- **AUC**: {better_metric_auc.capitalize()} distance performs better by {diff_auc:.4f}\n"
        
        if comparison[metric1]['f1_score'] > comparison[metric2]['f1_score']:
            better_metric_f1 = metric1
            diff_f1 = comparison[metric1]['f1_score'] - comparison[metric2]['f1_score']
        else:
            better_metric_f1 = metric2
            diff_f1 = comparison[metric2]['f1_score'] - comparison[metric1]['f1_score']
        
        report += f"- **F1-Score**: {better_metric_f1.capitalize()} distance performs better by {diff_f1:.4f}\n"
        
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

def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    
    return np.trapz(tpr, fpr)

def plot_roc_curves(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]], save_path: str = None) -> plt.Figure:
    
    logger.info("Plotting ROC curves")
    
    plt.figure(figsize=(10, 8))
    
    colors = {
        'euclidean': 'blue',
        'cosine': 'red'
    }
    
    first_key = next(iter(results['probabilities']))
    first_fold = results['probabilities'][first_key][0]
    y_true_first, y_proba_first = first_fold
    n_classes = y_proba_first.shape[1]
    for metric, (k, _) in optimal_k.items():
        key = f"{metric}_k{k}"
        
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        
        for fold, (y_true, y_proba) in enumerate(results['probabilities'][key]):
            roc_auc_results = calculate_multiclass_roc_auc(y_true, y_proba)
            fpr, tpr, _ = roc_auc_results['roc_curves']['micro']
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(results['probabilities'][key])
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        mean_auc = calculate_auc(mean_fpr, mean_tpr)
        plt.plot(
            mean_fpr, mean_tpr,
            color=colors[metric],
            label=f'{metric.capitalize()} (k={k}, AUC={mean_auc:.4f})',
            lw=2, alpha=0.8
        )
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, (metric, (k, _)) in enumerate(optimal_k.items()):
        key = f"{metric}_k{k}"
        avg_auc = np.mean(results['metrics']['auc'][key])
        plt.text(0.05, 0.05 + 0.05 * i, f'{metric.capitalize()} AUC: {avg_auc:.4f}',
                transform=plt.gca().transAxes, color=colors[metric], fontweight='bold')
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves plot saved to {save_path}")
    
    logger.info("ROC curves plotting completed")
    return plt.gcf()

def create_summary_report(stats: Dict[str, Any], analysis: Dict[str, Any], 
                         results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]], 
                         comparison: Dict[str, Dict[str, float]]) -> str:
   
    logger.info("Creating comprehensive summary report")
    
    report = "# Genetic Syndrome Classification Report\n\n"
    
    report += "## Executive Summary\n\n"
    report += "This report presents the analysis and classification of genetic syndrome embeddings derived from images. "
    report += "The embeddings are 320-dimensional vectors representing genetic syndrome characteristics. "
    report += "We performed data preprocessing, visualization using t-SNE, and classification using KNN with both Euclidean and Cosine distance metrics.\n\n"
    report += "### Key Findings\n\n"
    report += f"- The dataset contains {stats['total_images']} images from {stats['number_of_syndromes']} different genetic syndromes.\n"
    report += f"- Each syndrome has an average of {stats['images_per_syndrome']['mean']:.2f} images from {stats['subjects_per_syndrome']['mean']:.2f} subjects.\n"
    
    max_syndrome = max(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    min_syndrome = min(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    report += f"- The dataset shows some imbalance, with syndrome {max_syndrome[0]} having {max_syndrome[1]} images and syndrome {min_syndrome[0]} having only {min_syndrome[1]} images.\n"
    
    report += f"- t-SNE visualization revealed {analysis['number_of_clusters']} distinct clusters with {analysis['potential_overlap_count']} potential overlaps.\n"
    
    best_metric = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    best_metric_name = best_metric[0]
    best_k = optimal_k[best_metric_name][0]
    best_accuracy = best_metric[1]['accuracy']
    best_auc = best_metric[1]['auc']
    
    report += f"- The {best_metric_name.capitalize()} distance metric with k={best_k} performed best, achieving an accuracy of {best_accuracy:.4f} and AUC of {best_auc:.4f}.\n\n"
    
    report += "## Detailed Analysis\n\n"
    
    report += "### 1. Dataset Statistics\n\n"
    dataset_report = generate_dataset_statistics_report(stats)
    report += "\n".join(dataset_report.split("\n")[1:]) 
    report += "\n\n### 2. Clustering Analysis\n\n"
    clustering_report = generate_clustering_analysis_report(analysis)
    report += "\n".join(clustering_report.split("\n")[1:]) 
    report += "\n\n### 3. Cross-Validation Results\n\n"
    cv_report = generate_cross_validation_report(results, optimal_k)
    report += "\n".join(cv_report.split("\n")[1:])  
    report += "\n\n### 4. Distance Metrics Comparison\n\n"
    comparison_report = generate_comparison_report(comparison)
    report += "\n".join(comparison_report.split("\n")[1:])
    
    logger.info("Comprehensive summary report created")
    return report

def calculate_multiclass_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    classes = np.unique(y_true)
    n_classes = len(classes)
    y_bin = label_binarize(y_true, classes=classes)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    return {
        'roc_curves': {
            'class_wise': dict(zip(range(n_classes), zip(fpr.values(), tpr.values()))),
            'micro': (fpr["micro"], tpr["micro"], roc_auc["micro"])
        },
        'auc': {
            'class_wise': dict(zip(range(n_classes), roc_auc.values())),
            'micro': roc_auc["micro"]
        }
    }