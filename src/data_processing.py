import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Dict[str, Any]:
    try:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def flatten_data(data: Dict[str, Dict[str, Dict[str, List[float]]]]) -> pd.DataFrame:
    logger.info("Flattening hierarchical data structure")
    
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
    
    df = pd.DataFrame(rows)
    logger.info(f"Data flattened. Shape: {df.shape}")
    return df

flatten_hierarchical_data = flatten_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data")
    
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values found: {missing_values}")
        df = df.dropna()
        logger.info(f"Rows with missing values dropped. New shape: {df.shape}")
    
    embedding_lengths = df['embedding'].apply(len)
    if not (embedding_lengths == embedding_lengths.iloc[0]).all():
        logger.warning("Inconsistent embedding dimensions found")
        most_common_length = embedding_lengths.value_counts().idxmax()
        df = df[df['embedding'].apply(len) == most_common_length]
        logger.info(f"Filtered to embeddings with dimension {most_common_length}. New shape: {df.shape}")
    
    df['embedding'] = df['embedding'].apply(np.array)
    
    logger.info("Data preprocessing completed")
    return df

def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    logger.info("Generating dataset statistics")
    
    first_embedding = df['embedding'].iloc[0] if not df.empty else []
    embedding_dim = len(first_embedding) if isinstance(first_embedding, (list, np.ndarray)) else 0
    
    stats = {
        'total_images': len(df),
        'number_of_syndromes': df['syndrome_id'].nunique(),
        'number_of_subjects': df['subject_id'].nunique(),
        'embedding_dimension': embedding_dim,
        'syndrome_distribution': df['syndrome_id'].value_counts().to_dict(),
        'images_per_syndrome': df.groupby('syndrome_id').size().describe().to_dict(),
        'subjects_per_syndrome': df.groupby('syndrome_id')['subject_id'].nunique().describe().to_dict()
    }
    
    logger.info(f"Dataset statistics generated. Total images: {stats['total_images']}, Total syndromes: {stats['number_of_syndromes']}")
    return stats

def prepare_for_classification(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    logger.info("Preparing data for classification")
    
    X = np.stack(df['embedding'].values)
    
    syndrome_mapping = {}
    for i, syndrome in enumerate(df['syndrome_id'].unique()):
        if hasattr(syndrome, 'dtype') and np.issubdtype(syndrome.dtype, np.integer):
            syndrome_key = int(syndrome)
        else:
            try:
                syndrome_key = int(syndrome)
            except (TypeError, ValueError):
                syndrome_key = syndrome
                
        syndrome_mapping[i] = syndrome_key
    
    inverse_mapping = {syndrome: i for i, syndrome in syndrome_mapping.items()}
    
    y = np.array([inverse_mapping[df['syndrome_id'].iloc[i]] for i in range(len(df))])
    
    logger.info(f"Data prepared for classification. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, syndrome_mapping

prepare_data_for_classification = prepare_for_classification

def split_by_subject(df: pd.DataFrame) -> Dict[str, List[int]]:
    logger.info("Splitting data by subject")
    
    subject_indices = {}
    for syndrome_id, group in df.groupby('syndrome_id'):
        subject_indices[syndrome_id] = []
        for subject_id, subject_group in group.groupby('subject_id'):
            indices = subject_group.index.tolist()
            subject_indices[syndrome_id].append(indices)
    
    logger.info("Data split by subject completed")
    return subject_indices