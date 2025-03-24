# Relatório de Classificação de Síndromes Genéticas

## Resumo Executivo

Este relatório apresenta a análise e classificação de embeddings de síndromes genéticas derivados de imagens. Os embeddings são vetores de 320 dimensões que representam características de síndromes genéticas. Realizamos pré-processamento de dados, visualização usando t-SNE e classificação usando KNN com métricas de distância euclidiana e de cosseno.

### Principais Descobertas

- O conjunto de dados contém 60 imagens de 5 síndromes genéticas diferentes.
- Cada síndrome tem uma média de 12.00 imagens de 3.00 sujeitos.
- O conjunto de dados apresenta algum desequilíbrio, com a síndrome syndrome_001 tendo 12 imagens e a síndrome syndrome_001 tendo apenas 12 imagens.
- A visualização t-SNE revelou 5 clusters distintos com 0 sobreposições potenciais.
- A métrica de distância Euclidean com k=1 teve o melhor desempenho, alcançando uma precisão de 1.0000 e AUC de 1.0000.

## Análise Detalhada

### 1. Estatísticas do Conjunto de Dados


## Estatísticas Gerais
- Número total de imagens: 60
- Número de síndromes: 5
- Número de sujeitos: 15
- Dimensão dos embeddings: 320

## Estatísticas de Distribuição
### Imagens por Síndrome
- Média: 12.00
- Mínimo: 12
- Máximo: 12
- Desvio Padrão: 0.00

### Sujeitos por Síndrome
- Média: 3.00
- Mínimo: 3
- Máximo: 3
- Desvio Padrão: 0.00

## Distribuição de Síndromes
| ID da Síndrome | Número de Imagens |
| -------------- | ----------------- |
| syndrome_001 | 12 |
| syndrome_002 | 12 |
| syndrome_003 | 12 |
| syndrome_004 | 12 |
| syndrome_005 | 12 |

## Análise de Desequilíbrio de Dados
- O conjunto de dados parece razoavelmente equilibrado em termos de imagens por síndrome.


### 2. Análise de Agrupamentos


## Estatísticas Gerais
- Número de clusters: 5
- Sobreposições potenciais entre clusters: 0

## Relações entre Clusters
### Clusters Mais Próximos
- Par: ('syndrome_001', 'syndrome_005')
- Distância: 8.9715

### Clusters Mais Distantes
- Par: ('syndrome_001', 'syndrome_004')
- Distância: 28.3719

## Dispersões de Clusters
| Cluster (ID da Síndrome) | Dispersão |
| ------------------------- | --------- |
| syndrome_003 | 0.8127 |
| syndrome_001 | 0.8117 |
| syndrome_002 | 0.8104 |
| syndrome_005 | 0.7996 |
| syndrome_004 | 0.7540 |


### 3. Resultados da Validação Cruzada


## Valores Ótimos de k
| Métrica de Distância | k Ótimo | Precisão |
| --------------------- | ------- | -------- |
| Euclidean | 1 | 1.0000 |
| Cosine | 1 | 1.0000 |

## Métricas de Desempenho para Cada Configuração

### Métrica de Distância Euclidean
| k | Precisão | Precisão Top-k | F1-Score | AUC |
| - | -------- | -------------- | -------- | --- |
| **1** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### Métrica de Distância Cosine
| k | Precisão | Precisão Top-k | F1-Score | AUC |
| - | -------- | -------------- | -------- | --- |
| **1** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 13 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 14 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 15 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Análise de Tendências
- Para a métrica de distância **Euclidean**, há uma tendência de **estabilidade** na precisão à medida que k aumenta.
- Para a métrica de distância **Cosine**, há uma tendência de **estabilidade** na precisão à medida que k aumenta.


### 4. Comparação de Métricas de Distância


| Métrica de Distância | Precisão | Precisão Top-k | F1-Score | AUC |
| --------------------- | -------- | -------------- | -------- | --- |
| Euclidean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Cosine | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Análise Comparativa
- **Melhor Precisão**: Euclidean (1.0000)
- **Melhor Precisão Top-k**: Euclidean (1.0000)
- **Melhor F1-Score**: Euclidean (1.0000)
- **Melhor AUC**: Euclidean (1.0000)

### Diferenças de Desempenho
- **Diferença de Precisão**: 0.0000 (em favor de Cosine)
- **Diferença de Precisão Top-k**: 0.0000
- **Diferença de F1-Score**: 0.0000
- **Diferença de AUC**: 0.0000

### Recomendação
Com base nas métricas de avaliação, a métrica de distância **Euclidean** parece ter melhor desempenho geral para esta tarefa de classificação.

### Justificativa
A métrica Euclidean demonstrou:
- Maior precisão geral (1.0000)
- Melhor equilíbrio entre precisão e recall, com F1-Score de 1.0000
- Melhor poder discriminativo entre classes, com AUC de 1.0000
- Melhor desempenho em precisão Top-k (1.0000)


## Conclusão e Recomendações

Com base na análise abrangente realizada neste estudo, podemos tirar as seguintes conclusões:

1. A métrica de distância Euclidean superou as outras métricas para classificação KNN de síndromes genéticas.
2. O valor ótimo de k para o classificador KNN foi determinado como 1 através da validação cruzada.
3. A visualização t-SNE mostrou que algumas síndromes formam clusters distintos, enquanto outras têm sobreposição potencial, o que pode afetar o desempenho da classificação.

### Recomendações

1. **Seleção de Métricas de Distância**: Use a métrica de distância Euclidean para classificação KNN de embeddings de síndromes genéticas.
2. **Ajuste de Parâmetros**: Defina k=1 para desempenho ótimo no classificador KNN.
4. **Melhorias Adicionais**: Considere explorar outros algoritmos de classificação como SVM ou métodos ensemble para potencialmente melhorar o desempenho além da precisão alcançada de 1.0000.
5. **Engenharia de Características**: Investigue se a redução de dimensionalidade ou seleção de características poderia melhorar a classificação reduzindo o ruído nos embeddings.
