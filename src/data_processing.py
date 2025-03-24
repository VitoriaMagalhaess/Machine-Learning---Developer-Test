import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)
def load_data(file_path: str) -> Dict[str, Any]:
   
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        logger.warning(f"Erro ao carregar o arquivo com método padrão: {e}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding='bytes')
            logger.info("Arquivo carregado com encoding='bytes'")
            return data
        except Exception as e2:
            logger.warning(f"Erro ao carregar com encoding='bytes': {e2}")
            
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                logger.info("Arquivo carregado com encoding='latin1'")
                return data
            except Exception as e3:
                raise Exception(f"Falha ao carregar arquivo pickle. Tentativas esgotadas: {e}, {e2}, {e3}")
def flatten_data(data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> pd.DataFrame:
  
    rows = []
    
    for syndrome_id, subjects in data.items():
        for subject_id, images in subjects.items():
            for image_id, embedding in images.items():
                rows.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })
    
    return pd.DataFrame(rows)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
   
    na_count = df.isna().sum().sum()
    if na_count > 0:
        logger.warning(f"Encontrados {na_count} valores ausentes no conjunto de dados")

        df = df.dropna()
        logger.info(f"Linhas com valores ausentes removidas. Nova dimensão: {df.shape}")
    
    embedding_dimensions = [len(emb) for emb in df['embedding']]
    
    if len(set(embedding_dimensions)) > 1:
        logger.warning("Embeddings têm dimensões inconsistentes")
        
        from collections import Counter
        most_common_dim = Counter(embedding_dimensions).most_common(1)[0][0]
        logger.info(f"Dimensão mais comum: {most_common_dim}")
        
        df = df[df['embedding'].apply(lambda x: len(x) == most_common_dim)]
        logger.info(f"Filtrados embeddings para dimensão consistente. Nova dimensão: {df.shape}")
    
    if not isinstance(df.iloc[0]['embedding'], np.ndarray):
        logger.info("Convertendo embeddings para arrays numpy")
        df['embedding'] = df['embedding'].apply(lambda x: np.array(x, dtype=np.float32))
    
    return df
def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
   
    total_images = len(df)
    syndromes = df['syndrome_id'].unique()
    number_of_syndromes = len(syndromes)
    subjects = df['subject_id'].unique()
    number_of_subjects = len(subjects)
    embedding_dim = len(df.iloc[0]['embedding'])
    
    syndrome_counts = df['syndrome_id'].value_counts().to_dict()
    
    syndrome_df = pd.DataFrame(list(syndrome_counts.items()), columns=['syndrome_id', 'count'])
    
    images_per_syndrome = {
        'mean': syndrome_df['count'].mean(),
        'min': syndrome_df['count'].min(),
        'max': syndrome_df['count'].max(),
        'std': syndrome_df['count'].std()
    }
    
    subjects_per_syndrome = {}
    for syndrome in syndromes:
        subjects_per_syndrome[syndrome] = df[df['syndrome_id'] == syndrome]['subject_id'].nunique()
    
    subject_df = pd.DataFrame(list(subjects_per_syndrome.items()), columns=['syndrome_id', 'count'])
    
    subjects_per_syndrome_stats = {
        'mean': subject_df['count'].mean(),
        'min': subject_df['count'].min(),
        'max': subject_df['count'].max(),
        'std': subject_df['count'].std()
    }

    subject_to_syndromes = {}
    for _, row in df[['subject_id', 'syndrome_id']].drop_duplicates().iterrows():
        subject_id = row['subject_id']
        syndrome_id = row['syndrome_id']
        
        if subject_id not in subject_to_syndromes:
            subject_to_syndromes[subject_id] = []
        
        subject_to_syndromes[subject_id].append(syndrome_id)
    
    subjects_in_multiple_syndromes = [
        (subject_id, syndromes) 
        for subject_id, syndromes in subject_to_syndromes.items() 
        if len(syndromes) > 1
    ]
 
    stats = {
        'total_images': total_images,
        'number_of_syndromes': number_of_syndromes,
        'number_of_subjects': number_of_subjects,
        'embedding_dim': embedding_dim,
        'syndrome_distribution': syndrome_counts,
        'images_per_syndrome': images_per_syndrome,
        'subjects_per_syndrome_distribution': subjects_per_syndrome,
        'subjects_per_syndrome': subjects_per_syndrome_stats,
        'subjects_in_multiple_syndromes': subjects_in_multiple_syndromes,
        'has_subjects_in_multiple_syndromes': len(subjects_in_multiple_syndromes) > 0
    }
    
    return stats
def prepare_for_classification(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    
    X = np.array(df['embedding'].tolist())
    
    unique_syndromes = df['syndrome_id'].unique()
    syndrome_to_int = {syndrome: i for i, syndrome in enumerate(unique_syndromes)}
    int_to_syndrome = {i: syndrome for syndrome, i in syndrome_to_int.items()}
    
    y = np.array([syndrome_to_int[syndrome] for syndrome in df['syndrome_id']])
    
    return X, y, int_to_syndrome
def split_by_subject(df: pd.DataFrame) -> Dict[str, List[int]]:
    subject_indices = {}
    
    for syndrome_id in df['syndrome_id'].unique():
        syndrome_df = df[df['syndrome_id'] == syndrome_id]
        subject_indices[syndrome_id] = []
        
        for subject_id in syndrome_df['subject_id'].unique():
            indices = syndrome_df[syndrome_df['subject_id'] == subject_id].index.tolist()
            subject_indices[syndrome_id].append(indices)
    
    return subject_indices