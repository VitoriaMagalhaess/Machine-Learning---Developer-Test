#!/usr/bin/env python3
import numpy as np
import pickle
import os

def create_sample_data(output_file='data/sample_data.p', num_syndromes=5, num_subjects_per_syndrome=3, num_images_per_subject=4):
    """
    Cria um arquivo pickle com dados de amostra para testar o pipeline.
    Esta função cria uma estrutura similar à descrita no problema:
    {
        'syndrome_id': {
            'subject_id': {
                'image_id': [320-dimensional embedding]
            }
        }
    }
    
    Args:
        output_file: Caminho para salvar o arquivo pickle
        num_syndromes: Número de síndromes a serem geradas
        num_subjects_per_syndrome: Número de sujeitos por síndrome
        num_images_per_subject: Número de imagens por sujeito
    """
    print("Criando dados de amostra para testes...")
    
    # Cria o diretório de saída se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Estrutura de dados para armazenar os embeddings
    data = {}
    
    # Gera dados para cada síndrome
    for s in range(num_syndromes):
        syndrome_id = f'syndrome_{s+1:03d}'
        data[syndrome_id] = {}
        
        # Para cada síndrome, gera dados para cada sujeito
        for j in range(num_subjects_per_syndrome):
            subject_id = f'subject_{s+1:03d}_{j+1:03d}'
            data[syndrome_id][subject_id] = {}
            
            # Para cada sujeito, gera dados para cada imagem
            for k in range(num_images_per_subject):
                image_id = f'image_{s+1:03d}_{j+1:03d}_{k+1:03d}'
                
                # Gera um embedding de 320 dimensões com média diferente por síndrome
                # para criar alguma separabilidade nas classes
                mean_value = s * 0.5  # Média diferente para cada síndrome
                embedding = np.random.normal(mean_value, 1.0, 320).tolist()
                
                # Armazena o embedding
                data[syndrome_id][subject_id][image_id] = embedding
    
    # Salva os dados em um arquivo pickle
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    # Estatísticas dos dados gerados
    total_syndromes = len(data)
    total_subjects = sum(len(subjects) for subjects in data.values())
    total_images = sum(
        sum(len(images) for images in subjects.values()) 
        for subjects in data.values()
    )
    
    print(f"Dados de amostra criados com sucesso:")
    print(f"- Arquivo salvo em: {output_file}")
    print(f"- Total de síndromes: {total_syndromes}")
    print(f"- Total de sujeitos: {total_subjects}")
    print(f"- Total de imagens: {total_images}")
    print(f"- Dimensão dos embeddings: 320")
    print("\nAVISO: Estes são dados simulados apenas para fins de teste.")
    print("Para a análise real, você deve substituir este arquivo pelo arquivo de dados original.")

if __name__ == "__main__":
    create_sample_data()