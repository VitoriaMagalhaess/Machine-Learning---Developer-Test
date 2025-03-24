import os
import pickle
import numpy as np
import logging
from typing import Dict, List
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
def create_sample_data(output_file='data/sample_data.p', num_syndromes=5, num_subjects_per_syndrome=3, num_images_per_subject=4):
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    data = {}
    
    embedding_dim = 320
    
    np.random.seed(42)
    
    for i in range(1, num_syndromes + 1):
        syndrome_id = f"syndrome_{i:03d}"
        data[syndrome_id] = {}
        
        syndrome_mean = np.random.randn(embedding_dim)
        
        for j in range(1, num_subjects_per_syndrome + 1):
            subject_id = f"{i * 1000 + j}"
            data[syndrome_id][subject_id] = {}
            
            subject_mean = syndrome_mean + 0.5 * np.random.randn(embedding_dim)
            
            for k in range(1, num_images_per_subject + 1):
                image_id = f"{i * 10000 + j * 100 + k}"
                
                embedding = subject_mean + 0.2 * np.random.randn(embedding_dim)
                
                embedding = embedding / np.linalg.norm(embedding)
                
                data[syndrome_id][subject_id][image_id] = embedding
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    total_syndromes = len(data)
    total_subjects = sum(len(subjects) for subjects in data.values())
    total_images = sum(sum(len(images) for images in subjects.values()) for subjects in data.values())
    
    logger.info("Criando dados de amostra para testes...")
    logger.info("Dados de amostra criados com sucesso:")
    logger.info(f"- Arquivo salvo em: {output_file}")
    logger.info(f"- Total de síndromes: {total_syndromes}")
    logger.info(f"- Total de sujeitos: {total_subjects}")
    logger.info(f"- Total de imagens: {total_images}")
    logger.info(f"- Dimensão dos embeddings: {embedding_dim}")
    logger.info("")
    logger.info("AVISO: Estes são dados simulados apenas para fins de teste.")
    logger.info("Para a análise real, você deve substituir este arquivo pelo arquivo mini_gm_public_v0.1.p.")
if __name__ == "__main__":
    create_sample_data()