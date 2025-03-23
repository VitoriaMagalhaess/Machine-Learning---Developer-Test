import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> Dict[str, Any]:
    """
    Load data from pickle file.
    
    Args:
        file_path: Path to the pickle file.
        
    Returns:
        Dictionary containing the loaded data.
    """
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
    """
    Flatten the hierarchical data structure into a pandas DataFrame.
    
    Args:
        data: Hierarchical data structure with the format:
              {'syndrome_id': {'subject_id': {'image_id': [320-dimensional embedding]}}}
              
    Returns:
        DataFrame with columns: 'syndrome_id', 'subject_id', 'image_id', 'embedding'
    """
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

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by handling missing values and ensuring data integrity.
    
    Args:
        df: DataFrame with the flattened data.
        
    Returns:
        Preprocessed DataFrame.
    """
    logger.info("Preprocessing data")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Missing values found: {missing_values}")
        # Handle missing values - in this case, we'll drop rows with missing values
        df = df.dropna()
        logger.info(f"Rows with missing values dropped. New shape: {df.shape}")
    
    # Ensure all embeddings have the same dimension
    embedding_lengths = df['embedding'].apply(len)
    if not (embedding_lengths == embedding_lengths.iloc[0]).all():
        logger.warning("Inconsistent embedding dimensions found")
        # Filter out embeddings with inconsistent dimensions
        most_common_length = embedding_lengths.value_counts().idxmax()
        df = df[df['embedding'].apply(len) == most_common_length]
        logger.info(f"Filtered to embeddings with dimension {most_common_length}. New shape: {df.shape}")
    
    # Convert embeddings to numpy arrays for easier processing
    df['embedding'] = df['embedding'].apply(np.array)
    
    logger.info("Data preprocessing completed")
    return df

def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate statistics about the dataset.
    
    Args:
        df: DataFrame with the preprocessed data.
        
    Returns:
        Dictionary containing statistics about the dataset.
    """
    logger.info("Generating dataset statistics")
    
    stats = {
        'total_images': len(df),
        'number_of_syndromes': df['syndrome_id'].nunique(),
        'number_of_subjects': df['subject_id'].nunique(),
        'syndrome_distribution': df['syndrome_id'].value_counts().to_dict(),
        'images_per_syndrome': df.groupby('syndrome_id').size().describe().to_dict(),
        'subjects_per_syndrome': df.groupby('syndrome_id')['subject_id'].nunique().describe().to_dict(),
        'embedding_dimension': len(df['embedding'].iloc[0]) if not df.empty else 0
    }
    
    logger.info(f"Dataset statistics generated. Total images: {stats['total_images']}, Total syndromes: {stats['number_of_syndromes']}")
    return stats

def prepare_for_classification(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare the data for classification by extracting features (X) and labels (y).
    
    Args:
        df: DataFrame with the preprocessed data.
        
    Returns:
        Tuple containing features (X) and labels (y).
    """
    logger.info("Preparing data for classification")
    
    # Convert embeddings to a 2D array
    X = np.stack(df['embedding'].values)
    
    # Convert syndrome_ids to integer labels
    syndrome_mapping = {syndrome: i for i, syndrome in enumerate(df['syndrome_id'].unique())}
    df['syndrome_label'] = df['syndrome_id'].map(syndrome_mapping)
    y = df['syndrome_label'].values
    
    logger.info(f"Data prepared for classification. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, syndrome_mapping

def split_by_subject(df: pd.DataFrame) -> Dict[str, List[int]]:
    """
    Split data by subject to ensure proper cross-validation 
    (subjects should not be split across train/test).
    
    Args:
        df: DataFrame with the preprocessed data.
        
    Returns:
        Dictionary mapping syndrome_id to lists of subject indices.
    """
    logger.info("Splitting data by subject")
    
    # Group by syndrome_id and subject_id
    subject_indices = {}
    for syndrome_id, group in df.groupby('syndrome_id'):
        subject_indices[syndrome_id] = []
        for subject_id, subject_group in group.groupby('subject_id'):
            # Get the indices for this subject
            indices = subject_group.index.tolist()
            subject_indices[syndrome_id].append(indices)
    
    logger.info("Data split by subject completed")
    return subject_indices
