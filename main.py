import os
import argparse
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from src.data_processing import (
    load_data, flatten_data, preprocess_data,
    get_dataset_statistics, prepare_for_classification
)

from src.visualization import (
    apply_tsne, visualize_embeddings, analyze_clustering
)

from src.classification import (
    cross_validate, find_optimal_k, compare_distance_metrics
)

from src.reporting import (
    generate_dataset_statistics_report, generate_clustering_analysis_report,
    generate_cross_validation_report, generate_comparison_report,
    plot_roc_curves, create_summary_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def main(args):
    start_time = time()
    logger.info("Iniciando pipeline de classificação de síndromes genéticas")
    
    os.makedirs("output", exist_ok=True)
    
    logger.info("Passo 1: Processamento de dados")
    
    logger.info(f"Carregando dados de {args.data_file}")
    try:
        data = load_data(args.data_file)
        logger.info("Dados carregados com sucesso")
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        raise
    
    logger.info("Convertendo estrutura hierárquica em formato plano")
    df = flatten_data(data)
    logger.info(f"Estrutura convertida. Dimensões: {df.shape}")
    
    logger.info("Realizando pré-processamento dos dados")
    df = preprocess_data(df)
    logger.info("Pré-processamento concluído")
    
    logger.info("Gerando estatísticas do conjunto de dados")
    stats = get_dataset_statistics(df)
    logger.info(f"Estatísticas geradas. Total de imagens: {stats['total_images']}, Total de síndromes: {stats['number_of_syndromes']}")
    
    logger.info("Gerando relatório de estatísticas")
    stats_report = generate_dataset_statistics_report(stats)
    logger.info("Relatório de estatísticas gerado")
    
    if args.verbose:
        print("\n" + stats_report + "\n")
    
    with open("output/dataset_statistics_report.md", "w") as f:
        f.write(stats_report)
    
    logger.info("Passo 2: Visualização de dados")
    
    logger.info("Preparando dados para classificação")
    X, y, label_mapping = prepare_for_classification(df)
    logger.info(f"Dados preparados. Dimensões de X: {X.shape}, Dimensões de y: {y.shape}")
    
    perplexity = min(30, len(y) // 5)  
    logger.info(f"Aplicando t-SNE com perplexidade={perplexity}, n_iter=2000")
    reduced_embeddings = apply_tsne(X, perplexity=perplexity, n_iter=2000)
    logger.info(f"t-SNE concluído. Dimensões de saída: {reduced_embeddings.shape}")
    
    logger.info("Gerando visualização t-SNE")
    fig, ax = visualize_embeddings(
        reduced_embeddings, y, label_mapping,
        title="Visualização t-SNE dos Embeddings por Síndrome",
        save_path="output/tsne_visualization.png"
    )
    logger.info("Visualização t-SNE concluída")
    
    logger.info("Analisando agrupamentos de embeddings")
    cluster_analysis = analyze_clustering(reduced_embeddings, y, label_mapping)
    logger.info(f"Análise de agrupamentos concluída. Encontrados {cluster_analysis['number_of_clusters']} clusters com {cluster_analysis['potential_overlap_count']} sobreposições potenciais.")

    logger.info("Gerando relatório de análise de agrupamentos")
    clustering_report = generate_clustering_analysis_report(cluster_analysis)
    logger.info("Relatório de análise de agrupamentos gerado")
    
    if args.verbose:
        print("\n" + clustering_report + "\n")
    
    with open("output/clustering_analysis_report.md", "w") as f:
        f.write(clustering_report)
    
    logger.info("Passo 3: Tarefa de Classificação")
    
    k_values = list(range(1, 16))
    distance_metrics = ['euclidean', 'cosine']  
    n_folds = 10  
    
    logger.info(f"Iniciando validação cruzada com {n_folds} folds para valores de k {k_values} e métricas {distance_metrics}")
    results = cross_validate(X, y, k_values, distance_metrics, n_folds)
    logger.info("Validação cruzada concluída")
    
    logger.info("Avaliando resultados da validação cruzada")
    from src.metrics import evaluate_cross_validation_results
    results = evaluate_cross_validation_results(results)
    logger.info("Avaliação da validação cruzada concluída")
    
    logger.info("Determinando valores ótimos de k")
    optimal_k = find_optimal_k(results)
    for metric, (k, acc) in optimal_k.items():
        logger.info(f"Valor ótimo de k para distância {metric}: k={k} com precisão {acc:.4f}")
    
    logger.info("Comparando métricas de distância")
    comparison = compare_distance_metrics(results, optimal_k)
    logger.info("Comparação de métricas de distância concluída")
    
    logger.info("Gerando relatório de validação cruzada")
    cv_report = generate_cross_validation_report(results, optimal_k)
    logger.info("Relatório de validação cruzada gerado")
    
    if args.verbose:
        print("\n" + cv_report + "\n")
    
    with open("output/cross_validation_report.md", "w") as f:
        f.write(cv_report)
    
    logger.info("Gerando relatório de comparação de métricas de distância")
    comparison_report = generate_comparison_report(comparison)
    logger.info("Relatório de comparação de métricas de distância gerado")
    
    if args.verbose:
        print("\n" + comparison_report + "\n")
    
    with open("output/comparison_report.md", "w") as f:
        f.write(comparison_report)
    
    logger.info("Passo 4: Visualização dos Resultados")
    
    logger.info("Gerando curvas ROC")
    roc_fig = plot_roc_curves(results, optimal_k, save_path="output/roc_curves.png")
    logger.info("Geração de curvas ROC concluída")
    
    logger.info("Passo 5: Geração de Relatório Abrangente")
    
    summary_report = create_summary_report(
        stats, cluster_analysis, results, optimal_k, comparison
    )
    
    with open("output/summary_report.md", "w") as f:
        f.write(summary_report)
    
    end_time = time()
    execution_time = end_time - start_time
    
    logger.info(f"Pipeline de classificação de síndromes genéticas concluído em {execution_time:.2f} segundos")
    logger.info(f"Todos os resultados foram salvos no diretório {os.path.abspath('output')}")
    
    if args.verbose:
        print(f"\nTempo total de execução: {execution_time:.2f} segundos")
        print(f"Todos os resultados foram salvos em: {os.path.abspath('output')}")
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Classificação de Síndromes Genéticas")
    parser.add_argument("--data_file", type=str, default="data/mini_gm_public_v0.1.p",
                        help="Caminho para o arquivo pickle contendo os dados de embeddings")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar relatórios detalhados no console")
    args = parser.parse_args()
    
    os.makedirs("output", exist_ok=True)
    
    main(args)
