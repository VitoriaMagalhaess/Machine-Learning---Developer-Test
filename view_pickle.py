import pickle
import argparse
import numpy as np
import os

def view_pickle(pickle_file):
    """
    Exibe a estrutura e informações básicas de um arquivo pickle.
    
    Args:
        pickle_file: Caminho para o arquivo pickle
    """
    try:
        print(f"Tentando abrir: {pickle_file}")
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f'\nTipo de dados: {type(data)}')
        
        if isinstance(data, dict):
            print(f'Número de chaves no dicionário: {len(data)}')
            
            print('\nEstrutura do dicionário:')
            for i, (key, value) in enumerate(data.items()):
                if i >= 5:
                    print('...')
                    break
                print(f'Chave: {key}, Tipo: {type(value)}')
                
                if isinstance(value, dict):
                    subkeys = list(value.keys())
                    print(f'  Número de subchaves: {len(subkeys)}')
                    for j, subkey in enumerate(subkeys[:3]):
                        subvalue = value[subkey]
                        print(f'  Subchave: {subkey}, Tipo: {type(subvalue)}')
                        
                        if isinstance(subvalue, dict):
                            subsubkeys = list(subvalue.keys())
                            print(f'    Número de sub-subchaves: {len(subsubkeys)}')
                            for k, subsubkey in enumerate(subsubkeys[:3]):
                                subsubvalue = subvalue[subsubkey]
                                print(f'    Sub-subchave: {subsubkey}, Tipo: {type(subsubvalue)}')
                                
                                if isinstance(subsubvalue, (list, np.ndarray)):
                                    print(f'      Dimensão: {len(subsubvalue)}')
                                    print(f'      Primeiros elementos: {subsubvalue[:5]}...')
                        
                        if j >= 2:
                            print('  ...')
                            break
        elif isinstance(data, (list, np.ndarray)):
            print(f'\nComprimento: {len(data)}')
            print(f'Primeiros elementos: {data[:5]}')
            
        print("\nAnálise concluída!")
        
    except FileNotFoundError:
        print(f'Erro: Arquivo {pickle_file} não encontrado!')
    except Exception as e:
        print(f'Erro ao processar arquivo pickle: {str(e)}')


def create_dummy_data(output_file):
    """
    Cria um arquivo pickle de exemplo com estrutura semelhante ao esperado para o projeto.
    Este arquivo é apenas para teste e não representa dados reais.
    
    Args:
        output_file: Caminho para salvar o arquivo pickle
    """
    data = {}
    
    for syndrome_id in ['syndrome_001', 'syndrome_002', 'syndrome_003']:
        data[syndrome_id] = {}
    
        for subject_id in [f'subject_{i:03d}' for i in range(1, 4)]:
            data[syndrome_id][subject_id] = {}
        
            for image_id in [f'image_{i:03d}' for i in range(1, 4)]:
               
                embedding = np.random.randn(320).tolist()
                data[syndrome_id][subject_id][image_id] = embedding
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f'Arquivo de exemplo criado: {output_file}')
    print('Este arquivo contém dados simulados e é apenas para teste.')
    print('Para a análise real, você precisa substituí-lo pelo arquivo verdadeiro.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ferramenta para visualizar e trabalhar com arquivos pickle de embeddings.')
 
    subparsers = parser.add_subparsers(dest='command', help='Comando a ser executado')
    
    view_parser = subparsers.add_parser('view', help='Visualiza a estrutura de um arquivo pickle')
    view_parser.add_argument('pickle_file', help='Caminho para o arquivo pickle')
    
    dummy_parser = subparsers.add_parser('create_dummy', help='Cria um arquivo pickle de exemplo para testes')
    dummy_parser.add_argument('output_file', help='Caminho para salvar o arquivo pickle de exemplo')
    
    args = parser.parse_args()
    
    if args.command == 'view':
        view_pickle(args.pickle_file)
    elif args.command == 'create_dummy':
        create_dummy_data(args.output_file)
    else:
        parser.print_help()
        print("\nExemplos de uso:")
        print("  Para visualizar um arquivo pickle:")
        print("    python view_pickle.py view data/mini_gm_public_v0.1.p")
        print("  Para criar um arquivo de exemplo para testes:")
        print("    python view_pickle.py create_dummy data/example_data.p")
        import pickle
        
def open_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

data = open_pickle_file('data/mini_gm_public_v0.1.p')
print("Estrutura de dados carregada com sucesso.")
print(f"Total de síndromes: {len(data)}")