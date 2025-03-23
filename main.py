import os
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple

from src.data_processing import (
    load_data, flatten_data, preprocess_data, get_dataset_statistics,
    prepare_for_classification, split_by_subject
)
from src.visualization import (
    apply_tsne, visualize_embeddings, analyze_clustering
)
from src.classification import (
    cross_validate, find_optimal_k, compare_distance_metrics
)
from src.metrics import (
    evaluate_cross_validation_results
)
from src.reporting import (
    generate_dataset_statistics_report, generate_clustering_analysis_report,
    generate_cross_validation_report, generate_comparison_report,
    plot_roc_curves, create_summary_report
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    """
    Main function that orchestrates the entire analysis pipeline.
    
    Args:
        args: Command-line arguments.
    """
    logger.info("Starting genetic syndrome classification pipeline")
  
    logger.info("Step 1: Data Processing")
    
    data = load_data(args.data_file)
  
    df = flatten_data(data)
  
    df = preprocess_data(df)
    
    stats = get_dataset_statistics(df)
 
    dataset_report = generate_dataset_statistics_report(stats)
    if args.verbose:
        print("\n" + dataset_report)
    
    os.makedirs("output", exist_ok=True)
    with open("output/dataset_statistics_report.md", "w") as f:
        f.write(dataset_report)
    
    logger.info("Step 2: Data Visualization")
  
    X, y, syndrome_mapping = prepare_for_classification(df)
   
    reduced_embeddings = apply_tsne(X, perplexity=min(30, len(X) // 5), n_iter=2000)
   
    fig, ax = visualize_embeddings(
        reduced_embeddings, 
        y, 
        {v: k for k, v in syndrome_mapping.items()},
        save_path="output/tsne_visualization.png"
    )
   
    cluster_analysis = analyze_clustering(
        reduced_embeddings, 
        y, 
        {v: k for k, v in syndrome_mapping.items()}
    )

    clustering_report = generate_clustering_analysis_report(cluster_analysis)
    if args.verbose:
        print("\n" + clustering_report)
    
    with open("output/clustering_analysis_report.md", "w") as f:
        f.write(clustering_report)
    
    logger.info("Step 3: Classification Task")
    
    k_values = list(range(1, 16))
    
    distance_metrics = ['euclidean', 'cosine']
    
    cv_results = cross_validate(X, y, k_values, distance_metrics, n_folds=10)

    results = evaluate_cross_validation_results(cv_results)
  
    optimal_k = find_optimal_k(results)
    
    comparison = compare_distance_metrics(results, optimal_k)
    
    cv_report = generate_cross_validation_report(results, optimal_k)
    if args.verbose:
        print("\n" + cv_report)
    
    with open("output/cross_validation_report.md", "w") as f:
        f.write(cv_report)
 
    comparison_report = generate_comparison_report(comparison)
    if args.verbose:
        print("\n" + comparison_report)
   
    with open("output/comparison_report.md", "w") as f:
        f.write(comparison_report)

    logger.info("Step 4: Visualization of Results")
    
    roc_fig = plot_roc_curves(results, optimal_k, save_path="output/roc_curves.png")
    
    logger.info("Step 5: Generate Comprehensive Report")
    
    summary_report = create_summary_report(stats, cluster_analysis, results, optimal_k, comparison)
    if args.verbose:
        print("\n" + summary_report)

    with open("output/summary_report.md", "w") as f:
        f.write(summary_report)
    
    logger.info("Genetic syndrome classification pipeline completed")
    logger.info(f"All outputs saved to {os.path.abspath('output')} directory")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Syndrome Classification")
    parser.add_argument("--data_file", type=str, default="mini_gm_public_v0.1.p",
                        help="Path to the pickle file containing embeddings data")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed reports to console")
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    main(args)
