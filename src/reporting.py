import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
logger = logging.getLogger(__name__)
def generate_dataset_statistics_report(stats: Dict[str, Any]) -> str:

    report += "## Estatísticas Gerais\n"
    report += f"- Número total de imagens: {stats['total_images']}\n"
    report += f"- Número de síndromes: {stats['number_of_syndromes']}\n"
    report += f"- Número de sujeitos: {stats['number_of_subjects']}\n"
    report += f"- Dimensão dos embeddings: {stats['embedding_dim']}\n\n"
    
    report += "## Estatísticas de Distribuição\n"
    
    report += "### Imagens por Síndrome\n"
    report += f"- Média: {stats['images_per_syndrome']['mean']:.2f}\n"
    report += f"- Mínimo: {stats['images_per_syndrome']['min']}\n"
    report += f"- Máximo: {stats['images_per_syndrome']['max']}\n"
    report += f"- Desvio Padrão: {stats['images_per_syndrome']['std']:.2f}\n\n"
    
    report += "### Sujeitos por Síndrome\n"
    report += f"- Média: {stats['subjects_per_syndrome']['mean']:.2f}\n"
    report += f"- Mínimo: {stats['subjects_per_syndrome']['min']}\n"
    report += f"- Máximo: {stats['subjects_per_syndrome']['max']}\n"
    report += f"- Desvio Padrão: {stats['subjects_per_syndrome']['std']:.2f}\n\n"
    
    report += "## Distribuição de Síndromes\n"
    report += "| ID da Síndrome | Número de Imagens |\n"
    report += "| -------------- | ----------------- |\n"
    
    sorted_syndromes = sorted(
        stats['syndrome_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for syndrome, count in sorted_syndromes:
        report += f"| {syndrome} | {count} |\n"
    
    report += "\n## Análise de Desequilíbrio de Dados\n"
    
    if stats['images_per_syndrome']['std'] > 0.5 * stats['images_per_syndrome']['mean']:
        report += "- **Aviso**: O conjunto de dados apresenta desequilíbrio significativo na distribuição de imagens por síndrome.\n"
        
        imbalance_ratio = stats['images_per_syndrome']['max'] / max(1, stats['images_per_syndrome']['min'])
        report += f"- Razão de desequilíbrio (máx/mín): {imbalance_ratio:.2f}\n"
        
        report += "- **Sugestão**: Considere técnicas como amostragem ponderada, SMOTE, ou ajuste de hiperparâmetros para lidar com o desequilíbrio.\n"
    else:
        report += "- O conjunto de dados parece razoavelmente equilibrado em termos de imagens por síndrome.\n"
    
    logger.info("Relatório de estatísticas do conjunto de dados gerado")
    return report
def generate_clustering_analysis_report(analysis: Dict[str, Any]) -> str:
    
    report += "## Estatísticas Gerais\n"
    report += f"- Número de clusters: {analysis['number_of_clusters']}\n"
    report += f"- Sobreposições potenciais entre clusters: {analysis['potential_overlap_count']}\n\n"
    
    report += "## Relações entre Clusters\n"
    
    if analysis['closest_clusters']:
        closest_pair, closest_distance = analysis['closest_clusters']
        report += "### Clusters Mais Próximos\n"
        report += f"- Par: {closest_pair}\n"
        report += f"- Distância: {closest_distance:.4f}\n\n"
    
    if analysis['furthest_clusters']:
        furthest_pair, furthest_distance = analysis['furthest_clusters']
        report += "### Clusters Mais Distantes\n"
        report += f"- Par: {furthest_pair}\n"
        report += f"- Distância: {furthest_distance:.4f}\n\n"
    
    report += "## Dispersões de Clusters\n"
    report += "| Cluster (ID da Síndrome) | Dispersão |\n"
    report += "| ------------------------- | --------- |\n"
    
    sorted_dispersions = sorted(
        analysis['cluster_dispersions'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for cluster, dispersion in sorted_dispersions:
        report += f"| {cluster} | {dispersion:.4f} |\n"
    
    if analysis['potential_overlaps']:
        report += "\n## Sobreposições Potenciais\n"
        report += "As seguintes síndromes podem ter sobreposições em seus embeddings, o que pode afetar o desempenho da classificação:\n\n"
        
        for pair, distance in analysis['potential_overlaps']:
            report += f"- {pair[0]} e {pair[1]} (distância: {distance:.4f})\n"
    
    logger.info("Relatório de análise de agrupamentos gerado")
    return report
def generate_cross_validation_report(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]]) -> str:
   
    report = "# Relatório de Validação Cruzada\n\n"
    
    report += "## Valores Ótimos de k\n"
    report += "| Métrica de Distância | k Ótimo | Precisão |\n"
    report += "| --------------------- | ------- | -------- |\n"
    
    for metric, (k, accuracy) in optimal_k.items():
        report += f"| {metric.capitalize()} | {k} | {accuracy:.4f} |\n"
    
    report += "\n## Métricas de Desempenho para Cada Configuração\n"
    
    for metric in set(key.split('_')[0] for key in results['metrics']['accuracy'].keys()):
        report += f"\n### Métrica de Distância {metric.capitalize()}\n"
        report += "| k | Precisão | Precisão Top-k | F1-Score | AUC |\n"
        report += "| - | -------- | -------------- | -------- | --- |\n"
        
        for k in range(1, 16):
            key = f"{metric}_k{k}"
            
            if key in results['metrics']['accuracy']:
                accuracy = results['metrics']['accuracy'][key]
                top_k_accuracy = results['metrics']['top_k_accuracy'][key]
                f1 = results['metrics']['f1'][key]
                auc = results['metrics']['auc'][key]
                
                if optimal_k[metric][0] == k:
                    report += f"| **{k}** | **{accuracy:.4f}** | **{top_k_accuracy:.4f}** | **{f1:.4f}** | **{auc:.4f}** |\n"
                else:
                    report += f"| {k} | {accuracy:.4f} | {top_k_accuracy:.4f} | {f1:.4f} | {auc:.4f} |\n"
    
    report += "\n## Análise de Tendências\n"
    
    for metric in set(key.split('_')[0] for key in results['metrics']['accuracy'].keys()):
        accuracies = []
        for k in range(1, 16):
            key = f"{metric}_k{k}"
            if key in results['metrics']['accuracy']:
                accuracies.append(results['metrics']['accuracy'][key])
        
        if len(accuracies) >= 2:
            if accuracies[0] < accuracies[-1]:
                trend = "aumento"
            elif accuracies[0] > accuracies[-1]:
                trend = "diminuição"
            else:
                trend = "estabilidade"
            
            report += f"- Para a métrica de distância **{metric.capitalize()}**, há uma tendência de **{trend}** na precisão à medida que k aumenta.\n"
    
    logger.info("Relatório de validação cruzada gerado")
    return report
def generate_comparison_report(comparison: Dict[str, Dict[str, float]]) -> str:
    
    report = "# Relatório de Comparação de Métricas de Distância\n\n"
    
    report += "| Métrica de Distância | Precisão | Precisão Top-k | F1-Score | AUC |\n"
    report += "| --------------------- | -------- | -------------- | -------- | --- |\n"
    
    for metric, metrics in comparison.items():
        report += f"| {metric.capitalize()} | {metrics['accuracy']:.4f} | {metrics['top_k_accuracy']:.4f} | {metrics['f1']:.4f} | {metrics['auc']:.4f} |\n"
    
    report += "\n## Análise Comparativa\n"
    
    best_accuracy = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    best_top_k = max(comparison.items(), key=lambda x: x[1]['top_k_accuracy'])
    best_f1 = max(comparison.items(), key=lambda x: x[1]['f1'])
    best_auc = max(comparison.items(), key=lambda x: x[1]['auc'])
    
    report += f"- **Melhor Precisão**: {best_accuracy[0].capitalize()} ({best_accuracy[1]['accuracy']:.4f})\n"
    report += f"- **Melhor Precisão Top-k**: {best_top_k[0].capitalize()} ({best_top_k[1]['top_k_accuracy']:.4f})\n"
    report += f"- **Melhor F1-Score**: {best_f1[0].capitalize()} ({best_f1[1]['f1']:.4f})\n"
    report += f"- **Melhor AUC**: {best_auc[0].capitalize()} ({best_auc[1]['auc']:.4f})\n\n"
   
    if len(comparison) > 1:
        metrics_list = list(comparison.keys())
        metric1, metric2 = metrics_list[0], metrics_list[1]
        
        accuracy_diff = abs(comparison[metric1]['accuracy'] - comparison[metric2]['accuracy'])
        top_k_diff = abs(comparison[metric1]['top_k_accuracy'] - comparison[metric2]['top_k_accuracy'])
        f1_diff = abs(comparison[metric1]['f1'] - comparison[metric2]['f1'])
        auc_diff = abs(comparison[metric1]['auc'] - comparison[metric2]['auc'])
        
        better_metric_accuracy = metric1 if comparison[metric1]['accuracy'] > comparison[metric2]['accuracy'] else metric2
        
        report += "### Diferenças de Desempenho\n"
        report += f"- **Diferença de Precisão**: {accuracy_diff:.4f} (em favor de {better_metric_accuracy.capitalize()})\n"
        report += f"- **Diferença de Precisão Top-k**: {top_k_diff:.4f}\n"
        report += f"- **Diferença de F1-Score**: {f1_diff:.4f}\n"
        report += f"- **Diferença de AUC**: {auc_diff:.4f}\n\n"
    
    report += "### Recomendação\n"
    best_overall = max(comparison.items(), key=lambda x: (x[1]['accuracy'] + x[1]['auc'] + x[1]['f1'])/3)
    report += f"Com base nas métricas de avaliação, a métrica de distância **{best_overall[0].capitalize()}** parece ter melhor desempenho geral para esta tarefa de classificação.\n"
    
    report += "\n### Justificativa\n"
    report += f"A métrica {best_overall[0].capitalize()} demonstrou:\n"
    
    strengths = []
    if best_overall[0] == best_accuracy[0]:
        strengths.append(f"Maior precisão geral ({best_accuracy[1]['accuracy']:.4f})")
    if best_overall[0] == best_f1[0]:
        strengths.append(f"Melhor equilíbrio entre precisão e recall, com F1-Score de {best_f1[1]['f1']:.4f}")
    if best_overall[0] == best_auc[0]:
        strengths.append(f"Melhor poder discriminativo entre classes, com AUC de {best_auc[1]['auc']:.4f}")
    if best_overall[0] == best_top_k[0]:
        strengths.append(f"Melhor desempenho em precisão Top-k ({best_top_k[1]['top_k_accuracy']:.4f})")
    
    for strength in strengths:
        report += f"- {strength}\n"
    
    if not (best_overall[0] == best_accuracy[0] == best_f1[0] == best_auc[0] == best_top_k[0]):
        report += "\nApesar de não ser a melhor em todas as métricas individuais, a combinação de desempenho em todas as métricas a torna a escolha mais robusta para esta tarefa.\n"
    
    logger.info("Relatório de comparação de métricas de distância gerado")
    return report
def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:

    return np.trapz(tpr, fpr)
def plot_roc_curves(results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]], save_path: str = None) -> plt.Figure:
    
    logger.info("Gerando curvas ROC")
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.figure(figsize=(10, 8), dpi=150)
    
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
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Adivinhação Aleatória')
    
    plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
    plt.title('Curvas ROC (Receiver Operating Characteristic)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for i, (metric, (k, _)) in enumerate(optimal_k.items()):
        key = f"{metric}_k{k}"
        avg_auc = np.mean(results['metrics']['auc'][key])
        plt.text(0.05, 0.05 + 0.05 * i, f'{metric.capitalize()} AUC: {avg_auc:.4f}',
                transform=plt.gca().transAxes, color=colors[metric], fontweight='bold')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Curvas ROC salvas em {save_path}")
    
    logger.info("Geração de curvas ROC concluída")
    return plt.gcf()
def create_summary_report(stats: Dict[str, Any], analysis: Dict[str, Any], 
                         results: Dict[str, Any], optimal_k: Dict[str, Tuple[int, float]], 
                         comparison: Dict[str, Dict[str, float]]) -> str:
   
    logger.info("Criando relatório de resumo abrangente")
    
    report = "# Relatório de Classificação de Síndromes Genéticas\n\n"
    
    report += "## Resumo Executivo\n\n"
    report += "Este relatório apresenta a análise e classificação de embeddings de síndromes genéticas derivados de imagens. "
    report += "Os embeddings são vetores de 320 dimensões que representam características de síndromes genéticas. "
    report += "Realizei pré-processamento de dados, visualização usando t-SNE e classificação usando KNN com métricas de distância euclidiana e de cosseno.\n\n"
    
    report += "### Principais Descobertas\n\n"
    
    report += f"- O conjunto de dados contém {stats['total_images']} imagens de {stats['number_of_syndromes']} síndromes genéticas diferentes.\n"
    report += f"- Cada síndrome tem uma média de {stats['images_per_syndrome']['mean']:.2f} imagens de {stats['subjects_per_syndrome']['mean']:.2f} sujeitos.\n"
    
    max_syndrome = max(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    min_syndrome = min(stats['syndrome_distribution'].items(), key=lambda x: x[1])
    report += f"- O conjunto de dados apresenta algum desequilíbrio, com a síndrome {max_syndrome[0]} tendo {max_syndrome[1]} imagens e a síndrome {min_syndrome[0]} tendo apenas {min_syndrome[1]} imagens.\n"
    
    report += f"- A visualização t-SNE revelou {analysis['number_of_clusters']} clusters distintos com {analysis['potential_overlap_count']} sobreposições potenciais.\n"
    
    best_metric = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    best_metric_name = best_metric[0]
    best_k = optimal_k[best_metric_name][0]
    best_accuracy = best_metric[1]['accuracy']
    best_auc = best_metric[1]['auc']
    
    report += f"- A métrica de distância {best_metric_name.capitalize()} com k={best_k} teve o melhor desempenho, alcançando uma precisão de {best_accuracy:.4f} e AUC de {best_auc:.4f}.\n\n"
    
    report += "## Análise Detalhada\n\n"
    
    report += "### 1. Estatísticas do Conjunto de Dados\n\n"
    dataset_report = generate_dataset_statistics_report(stats)
    report += "\n".join(dataset_report.split("\n")[1:])  
    
    report += "\n\n### 2. Análise de Agrupamentos\n\n"
    clustering_report = generate_clustering_analysis_report(analysis)
    report += "\n".join(clustering_report.split("\n")[1:])  
    
    report += "\n\n### 3. Resultados da Validação Cruzada\n\n"
    cv_report = generate_cross_validation_report(results, optimal_k)
    report += "\n".join(cv_report.split("\n")[1:])  
    
    report += "\n\n### 4. Comparação de Métricas de Distância\n\n"
    comparison_report = generate_comparison_report(comparison)
    report += "\n".join(comparison_report.split("\n")[1:])  
    
    report += "\n\n## Conclusão e Recomendações\n\n"
    report += "Com base na análise abrangente realizada neste estudo, podemos tirar as seguintes conclusões:\n\n"
    
    report += f"1. A métrica de distância {best_metric_name.capitalize()} superou as outras métricas para classificação KNN de síndromes genéticas.\n"
    report += f"2. O valor ótimo de k para o classificador KNN foi determinado como {best_k} através da validação cruzada.\n"
    report += "3. A visualização t-SNE mostrou que algumas síndromes formam clusters distintos, enquanto outras têm sobreposição potencial, o que pode afetar o desempenho da classificação.\n\n"
    
    report += "### Recomendações\n\n"
    report += f"1. **Seleção de Métricas de Distância**: Use a métrica de distância {best_metric_name.capitalize()} para classificação KNN de embeddings de síndromes genéticas.\n"
    report += f"2. **Ajuste de Parâmetros**: Defina k={best_k} para desempenho ótimo no classificador KNN.\n"
    
    if stats['images_per_syndrome']['std'] > 0.5 * stats['images_per_syndrome']['mean']:
        report += "3. **Desequilíbrio de Dados**: Aborde o desequilíbrio no conjunto de dados coletando mais amostras para síndromes sub-representadas ou usando técnicas como SMOTE para geração de amostras sintéticas.\n"
    
    report += f"4. **Melhorias Adicionais**: Considere explorar outros algoritmos de classificação como SVM ou métodos ensemble para potencialmente melhorar o desempenho além da precisão alcançada de {best_accuracy:.4f}.\n"
    report += "5. **Engenharia de Características**: Investigue se a redução de dimensionalidade ou seleção de características poderia melhorar a classificação reduzindo o ruído nos embeddings.\n"
    
    logger.info("Relatório de resumo abrangente criado")
    return report
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