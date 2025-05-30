Repositório com a nova organização do repósitorio original [DeepIFSAC](https://github.com/mdsamad001/DeepIFSAC)

# 1. DeepIFSAC
This is the official implementation of paper titled, "DeepIFSAC: Deep Imputation of Missing Values Using Feature and Sample Attention within Contrastive Framework".

**DeepIFSAC** is a deep learning framework for tabular data that leverages attention-based architecture within a contrastive learning framework for missing value imputation. This repository provides code for data processing, training the DeepIFSAC model for missing value imputation on Tabular data set and a real-world EHR data set.

---

# 2. Overview

Esse código foi adaptado para realizar apenas missing value imputation (MVI), e por isso a entrada consiste em um pandas DataFrame (que pode ser carregado a partir de um arquivo `csv`, `tsv`, etc.) com apenas features numéricas (não é necessária a coluna com label). Na fase de treinamento utiliza CutMix para realizar data corruption e otimiza simultaneamente a constrative loss e a denoising loss. Após o treinamento realiza MVI no conjunto de treinamento e de teste (20% do dataset original), salvando os dados reconstruídos em arquivos CSV.

---

# 3. Instalação


## 3.1 Criar um virtual env com Conda

Se preferir utilizar Conda, é possivel criar um virtual env usando o arquivo YAML:

```bash
conda env create -f deepifsac_env.yml
conda activate deepifsac_env
```

## 3.2 Instalar usando pip

Caso prefira utilizar a instalação python nativa (se assegure de usar o Python 3.12.9), é possível instalar os pacotes utilizando pip:

```bash
pip install -r requirements.txt
```

---

## 4. Executando o código

## 4.1 Treinando o DeepIFSAC

No código `main.py` estão configuradas algumas variáveis globais, de acordo com o artigo do DeepIFSAC. Para realizar o treinamento, é necessário garantir que `FLAG_TRAIN = 1`. Também é importante validar os valores das variáveis globais `MISSING_TYPE` e `MISSING_RATE` estão adequadas para os dados de entrada. Para treinar, basta executar:

```bash
python main.py
```

---

## 5. Resultados e saídas

### 5.1 Pesos do modelo

Os pesos são salvos no diretório `./results/model_weights`. O nome do arquivo é composto pelo nome do dataset (variável `DATASET` em `main.py`), tipo de missing data (variável `MISSING_TYPE`), taxa de missing data (variável `MISSING_RATE`) e sufixo `cutmix` (data corruption).

### 5.2 Métricas

As métricas de treinamento (e.g., loss por época) são salvas como arquivos pickle no diretório `./results/training_scores` directory.

### 5.3 Dados reconstruídos

Ao final do treinamento, o modelo realiza MVI tanto no conjunto de treinamento quanto no de teste, salvando os arquivos `imputed_train_set.csv` e `imputed_test_set.csv` no diretório `./results/`.
A loss de reconstrução do conjunto de treinamento e a do conjunto de teste são impressas no terminal.

---

## 6. Estrutura do repositório

```
DeepIFSAC/
├── models
│   ├── deepifsac.py                # Implementation of the DeepIFSAC model.
│   ├── model.py                    # Additional models (e.g., simple_MLP).
│   └── ...
├── metrics
│   ├── loss.py                     # Implementation of loss functions.
├── missingness
│   ├── sampler.py                  # Sampler functions for data missing types.
│   ├── utils.py                    # Helper functions for sampling.
├── utils.py                        # Helper functions for training.
├── data_loader.py                  # Dataset and Dataloader classes.
├── corruptor.py                    # Corruptor class with different corruption methods (e.g. CutMix).
├── main.py                         # Main training code.
├── environment.yaml                # Environment configuration file.
└── README.md                       # This file.
```
