# Classificador Híbrido de Nível de Ansiedade Social

 Artur Daichi Gonçalves Inazaki - 14676716, Eduardo Veiga Moraes - 13633932,
 Pedro Henrique Chaves - 14835007, Rafael de Sousa Muniz - 14659644

## Visão Geral do Projeto

Este projeto implementa um modelo híbrido de Machine Learning, combinando uma Rede Neural Multilayer Perceptron (MLP) com Regressão Logística, para classificar o nível de ansiedade social a partir de um conjunto de dados abrangente. O objetivo principal é não apenas classificar, mas também fornecer insights sobre a importância das variáveis de entrada para cada branch do modelo e para o modelo híbrido como um todo, utilizando técnicas de análise de pesos e gradientes.

O modelo híbrido é projetado para aproveitar as capacidades de captura de relações não-lineares do MLP e a interpretabilidade da Regressão Logística. Diferentes métodos de combinação dos branches (soma ponderada, concatenação, atenção) são suportados para explorar a melhor forma de integrar suas previsões.

## Requisitos

Para executar este projeto, você precisará ter as seguintes bibliotecas Python instaladas. Você pode instalá-las via pip:

pip install numpy pandas torch scikit-learn matplotlib seaborn

##Como Usar

1. Organização
Deixe o código junto com o dataset em um mesmo diretório

2. Executar o Script
Abra seu terminal ou prompt de comando, navegue até o diretório onde você salvou ambos e use: 

python class_ans_soc.py enhanced_anxiety_dataset.csv
ou
python3 class_ans_soc.py enhanced_anxiety_dataset.csv

E

Com diferentes métodos de combinação:

Você pode especificar o método de combinação dos branches usando o argumento --combination:

weighted_sum (Padrão)

concatenation

attention

Como:

python class_ans_soc.py enhanced_anxiety_dataset.csv --combination concatenation
python class_ans_soc.py enhanced_anxiety_dataset.csv --combination attention
