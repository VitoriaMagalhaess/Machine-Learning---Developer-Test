import os
import sys
import pickle
import argparse
import numpy as np
from typing import Dict, Any, List
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)
def safe_load_pickle(pickle_file: str) -> Any:

    try:
        if not os.path.exists(pickle_file):
            logger.error(f"Arquivo não encontrado: {pickle_file}")
            return None
        with open(pickle_file, 'rb') as f:
            try:
                data = pickle.load(f)
                return data
            except Exception as e:
                logger.warning(f"Erro ao carregar com protocolo padrão: {e}")
                pass
        for protocol in [2, 3, 4, 5]:
            try:
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    logger.info(f"Arquivo carregado com sucesso usando protocolo alternativo.")
                    return data
            except Exception as e:
                logger.warning(f"Erro ao carregar com protocolo {protocol}: {e}")
                pass
        
        logger.error("Todas as tentativas de carregar o arquivo falharam.")
        return None
    
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar o arquivo: {e}")
        return None
def view_pickle(pickle_file: str, max_items: int = 5, max_depth: int = 3) -> None:

    logger.info(f"Carregando arquivo pickle: {pickle_file}")
    
    data = safe_load_pickle(pickle_file)
    
    if data is None:
        logger.error("Não foi possível carregar o arquivo. Verifique o formato e tente novamente.")
        return
    
    logger.info("Arquivo carregado com sucesso.")
    
    if isinstance(data, dict):
        num_syndromes = len(data)
        logger.info(f"Número de síndromes: {num_syndromes}")
        total_subjects = 0
        total_images = 0
        embedding_dims = set()
        logger.info("\nEstrutura do arquivo:")
        logger.info("--------------------")
        
        for i, (syndrome_id, subjects) in enumerate(data.items()):
            if i >= max_items and max_items > 0:
                logger.info(f"... e mais {num_syndromes - max_items} síndromes (omitidas para brevidade)")
                break
            
            num_subjects = len(subjects)
            total_subjects += num_subjects
            
            syndrome_images = 0
            
            logger.info(f"Síndrome: {syndrome_id}")
            logger.info(f"  Número de sujeitos: {num_subjects}")
            for j, (subject_id, images) in enumerate(subjects.items()):
                if j >= max_items and max_items > 0:
                    logger.info(f"  ... e mais {num_subjects - max_items} sujeitos (omitidos para brevidade)")
                    break
                
                num_images = len(images)
                syndrome_images += num_images
                
                logger.info(f"  Sujeito: {subject_id}")
                logger.info(f"    Número de imagens: {num_images}")
                for k, (image_id, embedding) in enumerate(images.items()):
                    if k >= max_items and max_items > 0:
                        logger.info(f"    ... e mais {num_images - max_items} imagens (omitidas para brevidade)")
                        break
                    
                    embedding_dims.add(len(embedding))
                    
                    logger.info(f"    Imagem: {image_id}")
                    logger.info(f"      Dimensão do embedding: {len(embedding)}")
                    logger.info(f"      Primeiros 5 valores: {embedding[:5]}")
                    logger.info(f"      Últimos 5 valores: {embedding[-5:]}")
            
            total_images += syndrome_images
            logger.info(f"  Total de imagens para esta síndrome: {syndrome_images}")
            logger.info("")
        logger.info("\nResumo Geral:")
        logger.info("-------------")
        logger.info(f"Total de síndromes: {num_syndromes}")
        logger.info(f"Total de sujeitos: {total_subjects}")
        logger.info(f"Total de imagens: {total_images}")
        
        if len(embedding_dims) == 1:
            logger.info(f"Dimensão dos embeddings: {next(iter(embedding_dims))}")
        else:
            logger.info(f"Dimensões dos embeddings (inconsistentes): {embedding_dims}")
        logger.info("\nValidação da Estrutura:")
        logger.info("----------------------")
        
        if num_syndromes > 0:
            logger.info("✓ Arquivo contém síndromes")
        else:
            logger.info("✗ Arquivo não contém síndromes")
        
        if total_subjects > 0:
            logger.info("✓ Síndromes contêm sujeitos")
        else:
            logger.info("✗ Síndromes não contêm sujeitos")
        
        if total_images > 0:
            logger.info("✓ Sujeitos contêm imagens")
        else:
            logger.info("✗ Sujeitos não contêm imagens")
        
        if len(embedding_dims) == 1:
            logger.info(f"✓ Todos os embeddings têm a mesma dimensão: {next(iter(embedding_dims))}")
        else:
            logger.info(f"✗ Embeddings têm dimensões inconsistentes: {embedding_dims}")
        
        logger.info("\nO arquivo parece estar no formato esperado para o projeto de classificação de síndromes genéticas.")
    else:
        logger.warning(f"O arquivo não está no formato de dicionário esperado. Formato atual: {type(data)}")
        logger.info("Tentando mostrar informações gerais sobre o conteúdo:")
        
        if hasattr(data, '__len__'):
            logger.info(f"Comprimento: {len(data)}")
        
        if hasattr(data, 'shape'):
            logger.info(f"Forma (shape): {data.shape}")
        
        if hasattr(data, 'dtypes'):
            logger.info(f"Tipos de dados: {data.dtypes}")
def create_dummy_data(output_file):
    from create_sample_data import create_sample_data
    create_sample_data(output_file=output_file)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador de arquivos pickle para o projeto de classificação de síndromes genéticas")
    parser.add_argument("--pickle_file", type=str, help="Caminho para o arquivo pickle a ser visualizado")
    parser.add_argument("--max_items", type=int, default=3, help="Número máximo de itens a serem exibidos em cada nível")
    parser.add_argument("--create_dummy", type=str, help="Criar um arquivo pickle de exemplo com a estrutura esperada")
    
    args = parser.parse_args()
    
    if args.create_dummy:
        create_dummy_data(args.create_dummy)
        logger.info(f"Arquivo de exemplo criado em: {args.create_dummy}")
    elif args.pickle_file:
        view_pickle(args.pickle_file, args.max_items)
    else:
        logger.error("Nenhuma operação especificada. Use --pickle_file para visualizar um arquivo ou --create_dummy para criar um arquivo de exemplo.")
        parser.print_help()