# MACHINE-LEARNING---DEVELOPER-TEST
# LISTA DE COMANDOS PARA EXECUÇÃO DO CÓDIGO

A seguir, apresenta-se um conjunto de comandos organizados para a correta execução do pipeline de análise, desde a configuração inicial do ambiente até a obtenção dos resultados finais.

## Configuração Inicial

 Inicialmente é necessário preparar o ambiente de trabalho.

## Criar Diretórios para Armazenamento de Dados e Resultados

mkdir -p data
mkdir -p output

### Objetivo: Criar as pastas "data" e "output" para armazenar os arquivos de entrada e saída, respectivamente.

## Instalar as Bibliotecas Necessárias

pip install numpy pandas scikit-learn matplotlib scipy tqdm

### Objetivo: Instalar todas as bibliotecas Python utilizadas no projeto, garantindo que o ambiente possua os pacotes necessários para a execução do código.

## Execução do Pipeline de Análise

Após a configuração inicial, deve-se proceder com a geração dos dados de amostra, análise dos dados e execução do modelo de classificação.

## Gerar Dados de Amostra

python create_sample_data.py

### Objetivo: Criar um arquivo pickle contendo um conjunto de dados simulados, composto por 60 embeddings de 5 síndromes, que será utilizado para testes e validação do sistema.

## Visualizar a Estrutura do Arquivo de Dados

python view_pickle.py --pickle_file data/sample_data.p

### Objetivo: Exibir a estrutura e o conteúdo do arquivo de dados, apresentando informações como síndromes, sujeitos e embeddings armazenados no dataset.

## Executar o Pipeline Completo de Análise

python main.py --data_file data/sample_data.p

### Objetivo:Executar todas as etapas do pipeline, incluindo:

Processamento de dados

Visualização de embeddings utilizando t-SNE

Classificação com K-Nearest Neighbors (KNN)

Geração de relatórios e métricas de avaliação

## Executar a Análise com Saída Detalhada (Opcional)

python main.py --data_file data/sample_data.p --verbose

### Objetivo: Executar o pipeline completo e exibir relatórios detalhados diretamente no terminal para um acompanhamento mais informativo.

## Análise dos Resultados

Após a execução do código, os resultados serão armazenados no diretório "output". A seguir, apresentam-se os comandos para acessar os arquivos gerados.

## Listar os Arquivos Gerados na Pasta de Saída

ls -la output/

### Objetivo: Exibir todos os arquivos presentes no diretório de saída, possibilitando a verificação dos resultados gerados.

## Verificar o Relatório Resumido da Análise

cat output/summary_report.md

### Objetivo: Apresentar um resumo geral da análise, incluindo as principais métricas e conclusões obtidas.

## Verificar Estatísticas do Conjunto de Dados

cat output/dataset_statistics_report.md

### Objetivo: Exibir estatísticas detalhadas sobre o dataset analisado, como distribuição de síndromes e quantidade de imagens por classe.

## Verificar os Resultados da Validação Cruzada

cat output/cross_validation_report.md

## Apresentar os resultados detalhados da validação cruzada realizada durante a classificação, incluindo métricas como AUC, F1-Score e Top-k Accuracy.

## Sequência Recomendada para Execução Completa

Para garantir a correta execução de todas as etapas do pipeline, recomenda-se seguir a seguinte ordem de comandos:

mkdir -p data output
pip install numpy pandas scikit-learn matplotlib scipy tqdm
python create_sample_data.py
python view_pickle.py --pickle_file data/sample_data.p
python main.py --data_file data/sample_data.p
ls -la output/
cat output/summary_report.md

## Essa sequência garante que:

As pastas necessárias sejam criadas.

As dependências sejam instaladas.

Um conjunto de dados de teste seja gerado.

A estrutura dos dados seja visualizada.

O pipeline completo de análise seja executado.

Os arquivos de saída sejam listados.

O relatório final seja acessado.