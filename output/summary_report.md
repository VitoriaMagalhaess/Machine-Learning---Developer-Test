# Genetic Syndrome Classification Report

## Executive Summary

This report presents the analysis and classification of genetic syndrome embeddings derived from images. The embeddings are 320-dimensional vectors representing genetic syndrome characteristics. We performed data preprocessing, visualization using t-SNE, and classification using KNN with both Euclidean and Cosine distance metrics.

### Key Findings

- The dataset contains 60 images from 5 different genetic syndromes.
- Each syndrome has an average of 12.00 images from 3.00 subjects.
- The dataset shows some imbalance, with syndrome syndrome_001 having 12 images and syndrome syndrome_001 having only 12 images.
- t-SNE visualization revealed 5 distinct clusters with 0 potential overlaps.
- The Euclidean distance metric with k=10 performed best, achieving an accuracy of 1.0000 and AUC of 1.0000.

## Detailed Analysis

### 1. Dataset Statistics


## General Statistics
- Total number of images: 60
- Number of syndromes: 5
- Number of subjects: 15
- Embedding dimension: 320

## Distribution Statistics
### Images per Syndrome
- Mean: 12.00
- Min: 12
- Max: 12
- Std: 0.00

### Subjects per Syndrome
- Mean: 3.00
- Min: 3
- Max: 3
- Std: 0.00

## Syndrome Distribution
| Syndrome ID | Number of Images |
| ----------- | ---------------- |
| syndrome_001 | 12 |
| syndrome_002 | 12 |
| syndrome_003 | 12 |
| syndrome_004 | 12 |
| syndrome_005 | 12 |


### 2. Clustering Analysis


## General Statistics
- Number of clusters: 5
- Potential cluster overlaps: 0

## Cluster Relationships
### Closest Clusters
- Pair: ('syndrome_001', 'syndrome_002')
- Distance: 5.8672

### Furthest Clusters
- Pair: ('syndrome_001', 'syndrome_005')
- Distance: 28.2329

## Cluster Dispersions
| Cluster (Syndrome ID) | Dispersion |
| --------------------- | ---------- |
| syndrome_004 | 1.9780 |
| syndrome_002 | 1.5728 |
| syndrome_003 | 1.5720 |
| syndrome_001 | 1.5048 |
| syndrome_005 | 1.4823 |


### 3. Cross-Validation Results


## Optimal k Values
| Distance Metric | Optimal k | Accuracy |
| --------------- | --------- | -------- |
| euclidean | 10 | 1.0000 |
| cosine | 1 | 0.3500 |

## Performance Metrics for Each Configuration

### Euclidean Distance Metric
| k | Accuracy | Top-k Accuracy | F1-Score | AUC |
| - | -------- | -------------- | -------- | --- |
| 1 | 0.9667 | 0.9667 | 0.9627 | 0.9715 |
| 2 | 0.9500 | 0.9833 | 0.9567 | 0.9765 |
| 3 | 0.9833 | 0.9833 | 0.9760 | 0.9790 |
| 4 | 0.9833 | 1.0000 | 0.9760 | 0.9975 |
| 5 | 0.9833 | 1.0000 | 0.9760 | 1.0000 |
| 6 | 0.9833 | 1.0000 | 0.9760 | 1.0000 |
| 7 | 0.9833 | 1.0000 | 0.9760 | 1.0000 |
| 8 | 0.9833 | 1.0000 | 0.9760 | 1.0000 |
| 9 | 0.9833 | 1.0000 | 0.9733 | 1.0000 |
| 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 11 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 12 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 13 | 0.9667 | 1.0000 | 0.9600 | 1.0000 |
| 14 | 0.9667 | 1.0000 | 0.9600 | 1.0000 |
| 15 | 0.9667 | 1.0000 | 0.9600 | 1.0000 |

### Cosine Distance Metric
| k | Accuracy | Top-k Accuracy | F1-Score | AUC |
| - | -------- | -------------- | -------- | --- |
| 1 | 0.3500 | 0.6167 | 0.2409 | 0.5580 |
| 2 | 0.3500 | 0.6167 | 0.2422 | 0.5660 |
| 3 | 0.3167 | 0.5833 | 0.1989 | 0.5740 |
| 4 | 0.3167 | 0.6667 | 0.1966 | 0.6060 |
| 5 | 0.3000 | 0.6667 | 0.1789 | 0.6060 |
| 6 | 0.2833 | 0.6167 | 0.1709 | 0.6140 |
| 7 | 0.3167 | 0.6333 | 0.1909 | 0.6270 |
| 8 | 0.3000 | 0.6500 | 0.1775 | 0.6320 |
| 9 | 0.3000 | 0.6167 | 0.1766 | 0.6370 |
| 10 | 0.3000 | 0.6500 | 0.1766 | 0.6410 |
| 11 | 0.3000 | 0.7500 | 0.1909 | 0.6850 |
| 12 | 0.3000 | 0.6833 | 0.1899 | 0.6890 |
| 13 | 0.3000 | 0.6833 | 0.1899 | 0.6755 |
| 14 | 0.2833 | 0.6833 | 0.1699 | 0.6570 |
| 15 | 0.2833 | 0.6333 | 0.1699 | 0.6400 |


### 4. Distance Metrics Comparison


| Distance Metric | Accuracy | Top-k Accuracy | F1-Score | AUC |
| --------------- | -------- | -------------- | -------- | --- |
| Euclidean | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Cosine | 0.3500 | 0.6167 | 0.2409 | 0.5580 |

## Comparison Analysis
- **Accuracy**: Euclidean distance performs better by 0.6500
- **AUC**: Euclidean distance performs better by 0.4420
- **F1-Score**: Euclidean distance performs better by 0.7591

### Recommendation
Based on the evaluation metrics, the **Euclidean distance** metric appears to perform better overall for this classification task.


## Conclusion and Recommendations

Based on the comprehensive analysis performed in this study, we can draw the following conclusions:

1. The Euclidean distance metric outperformed the other metrics for KNN classification of genetic syndromes.
2. The optimal value of k for the KNN classifier was determined to be 10 through cross-validation.
3. t-SNE visualization showed that some syndromes form distinct clusters, while others have potential overlap, which may affect classification performance.

### Recommendations

1. **Distance Metric Selection**: Use the Euclidean distance metric for KNN classification of genetic syndrome embeddings.
2. **Parameter Tuning**: Set k=10 for optimal performance in the KNN classifier.
4. **Further Improvements**: Consider exploring other classification algorithms like SVM or ensemble methods to potentially improve performance beyond the achieved accuracy of 1.0000.
5. **Feature Engineering**: Investigate whether dimensionality reduction or feature selection could improve classification by reducing noise in the embeddings.
