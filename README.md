<<<<<<< HEAD
# Sistema de Avaliação de Risco de Crédito
=======
# Risco ML - Sistema de Avaliação de Risco de Crédito 💳📊
>>>>>>> 69ddf42 (Criacao dos módulos para chamada do modelo e primeira versao do main.py)

## 🎯 Objetivo do Projeto

Desenvolver um modelo de machine learning para prever a probabilidade de **default** em empréstimos pessoais.  
O sistema permite classificar novos clientes em categorias de risco (**alto/baixo**) com base em características financeiras e históricas.

## 🏗️ Arquitetura e Organização

### Estrutura de Diretórios



```
<<<<<<< HEAD
SISTENA-AVALIACAORISCO-FINANCEIRO/
├── app/                           # Novo nome para streamlit_app/ (template)
│   └── streamlit_app.py                     # Aplicação web para fazer predições
=======
SISTEMA-AVALIACAORISCO-FINANCEIRO/
├── app/
│   └── streamlit_app.py
>>>>>>> 69ddf42 (Criacao dos módulos para chamada do modelo e primeira versao do main.py)
│
├── configs/
 (YAML, JSON, etc.)
│
├── data/
│   ├── raw/
│   │   └── credit_risk_dataset.csv
│   │
│   ├── interim/
│   │   └── dados_novos.csv
│   │
│   ├── processed/
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   └── y_test.pkl
│   │
│   └── scalers/
│       └── feature_scalers.pkl
treinamento
│
├── experiments/
│   ├── mlruns/                    # Histórico de execuções do MLflow
│   └── models/                    # Modelos registrados
│
├── notebooks/
│   ├── 1-EDA.ipynb                # Análise exploratória de dados
│   ├── 2-EngenhariaAtributos.ipynb# Criação e transformação de features
│   ├── 3-Preprocessamento.ipynb   # Limpeza e preparação dos dados
│   └── 4-Modelagem.ipynb          # Treinamento e otimização de modelos
│
├── src/                           # Código fonte do projeto
│   ├── features/
│   │   └── feature_store.py       # Carregamento e transformação de features em produção
│   │
│   ├── models/
│   │   ├── modelo_producao.py     # Interface para carregar modelo do MLflow
│   │   └── __init__.py
│   │
│   ├── pipeline/
│   │   ├── engenharia_atributos.py# Funções de feature engineering
│   │   ├── preprocessamento_dados.py# Funções de preprocessamento
│   │   ├── production_pipeline.py # Pipeline de produção
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── paths.py               # Gerenciamento de caminhos do projeto
│   │   └── __init__.py
│   │
│   └── __init__.py
│
├── tests/                         # Testes unitários e de integração
│
├── .env                           # Variáveis de ambiente
├── .gitignore                     # Arquivos e pastas a serem ignorados pelo Git
├── pyproject.toml                 # Dependências do projeto (alternativa ao requirements.txt)
└── README.md                      # Documentação do projeto
```


## 🔄 Fluxo de Dados



```
Dataset Bruto (data/raw/)
    ↓
EDA e Feature Engineering (notebooks/1 e 2)
    ↓
Preprocessamento (notebook 3)
    ↓
Treinamento de Modelos (notebook 4)
    ↓
Modelo Promovido em Production (MLflow Registry)
    ↓
Feature Store + Modelo (em produção)
    ↓
Aplicação Streamlit (Interface para Usuários)
```


## 🛠️ Tecnologias Utilizadas

### Processamento de Dados
- **pandas** 🐼: Manipulação e análise de dados
- **numpy** 🔢: Operações numéricas
- **scikit-learn** 🧠: Preprocessamento e métricas

### Modelagem
- **RandomForest, LogisticRegression, KNN, MLP** 🌳
- **xgboost** ⚡
- **lightgbm** 💡 (modelo final)
- **optuna** 🎯: Otimização de hiperparâmetros

### MLOps e Tracking
- **mlflow** 📋: Registro e versionamento de modelos

### Visualização
- **matplotlib** 📈
- **seaborn** 📊
- **plotly** 🌐
- **shap** 🔍: Explicabilidade do modelo

### Interface Web
- **streamlit** 💻: Aplicação web para predições

## 🚀 Como Usar

### Instalação de Dependências

```bash
pip install -e .

```

Todas as dependências estão definidas em `pyproject.toml`.


## Fluxo de Desenvolvimento

### Notebooks

Os notebooks estão organizados sequencialmente:

1. **1-EDA.ipynb**: Análise inicial dos dados, distribuições e correlações
2. **2-EngenhariaAtributos.ipynb**: Criação de novas variáveis e categorização
3. **3-Preprocessamento.ipynb**: Normalização, tratamento de valores faltantes e splitting
4. **4-Modelagem.ipynb**: Treinamento de múltiplos modelos, otimização e seleção do melhor

### Modelos Treinados

Durante o treinamento, os seguintes modelos são avaliados:

- RandomForestClassifier
- XGBClassifier
- LGBMClassifier (melhor performance)
- KNeighborsClassifier
- LogisticRegression
- MLPClassifier

O modelo LightGBM é otimizado com Optuna e promovido para Production.

## Configuração do MLflow

O projeto utiliza MLflow para tracking de experimentos e registro de modelos.

Opções para abrir a interface do MLflow (portável):

- Usando caminho relativo (execute a partir da raiz do projeto):

```bash
# roda o UI apontando para experiments/mlruns na raiz do projeto
mlflow ui --backend-store-uri file:experiments/mlruns --default-artifact-root file:experiments/mlruns
```

- Usando variável de ambiente (sessão ou permanente) — após setar `MLFLOW_TRACKING_URI` basta executar `mlflow ui`:

CMD (temporário na sessão):
```cmd
set MLFLOW_TRACKING_URI=file:///C:/caminho/para/o/projeto/experiments/mlruns
mlflow ui
```

PowerShell (temporário na sessão):
```powershell
$env:MLFLOW_TRACKING_URI = 'file:///C:/caminho/para/o/projeto/experiments/mlruns'
mlflow ui
```

Para setar permanentemente (Windows):
```cmd
setx MLFLOW_TRACKING_URI "file:///C:/caminho/para/o/projeto/experiments/mlruns"
```

- Script automatizado (recomendado): existe um script PowerShell em `scripts/start-mlflow.ps1` que resolve o caminho do repositório automaticamente e inicia o UI sem precisar editar caminhos pessoais.

Exemplo de uso (PowerShell, a partir da raiz do repo):
```powershell
# executa o script que monta a URI dinamicamente
.\scripts\start-mlflow.ps1
```

Observações:
- Se você configurar `MLFLOW_TRACKING_URI` na sua sessão, execute o `mlflow ui` no mesmo terminal onde a variável foi definida.
- Se o UI mostrar experimentos vazios mesmo com arquivos em `experiments/mlruns`, verifique/remova a variável `MLFLOW_TRACKING_URI` e reinicie o `mlflow ui` no terminal correto.


Acesse em `http://localhost:5000` para visualizar:
- Todos os experimentos executados
- Métricas de cada modelo
- Artefatos (gráficos, feature importance)
- Modelos registrados e seu stage (Production, Staging, etc)

## Métricas Utilizadas

- **Accuracy**: Proporção de acertos
- **Precision**: De todos os defaults previstos, quantos estão corretos
- **Recall**: De todos os defaults reais, quantos foram capturados
- **F1-Score**: Média harmônica entre precision e recall
- **ROC-AUC**: Área sob a curva ROC
- **PR-AUC**: Area Precision-Recall (métrica principal para otimização)

## Threshold de Decisão

O threshold (ponto de corte) foi otimizado utilizando a estatística KS (Kolmogorov-Smirnov):

- **Threshold padrão**: 0.42
- Probabilidade >= 0.42: Classificado como "Alto Risco"
- Probabilidade < 0.42: Classificado como "Baixo Risco"

## Dependências Principais

```
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
lightgbm>=4.3.0
xgboost>=2.0.3
optuna>=3.5.0
mlflow>=2.14.0
streamlit>=1.36.0
matplotlib>=3.8.0
seaborn>=0.13.0
shap>=0.44.1
```

Veja `pyproject.toml` para a lista completa.

## Estrutura de Dados de Entrada

Para fazer predições, os dados devem conter as seguintes colunas:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| person_age | int | Idade (20-95) |
| person_income | float | Renda anual em dólares |
| person_emp_length | float | Anos de emprego |
| loan_intent | str | Intenção do empréstimo |
| loan_grade | str | Grade do empréstimo (A-G) |
| loan_amnt | float | Valor do empréstimo |
| loan_int_rate | float | Taxa de juros |
| loan_percent_income | float | Empréstimo como % da renda |
| cb_person_default_on_file | str | Teve default anterior (Y/N) |
| cb_person_cred_history_length | int | Anos de histórico de crédito |
| loan_status | int | Status do empréstimo (0/1) |

## Próximos Passos

Melhorias futuras podem incluir:

- Utilização do docker
- Deploy na AWS
- Implementar testes unitários
<<<<<<< HEAD
=======

>>>>>>> 69ddf42 (Criacao dos módulos para chamada do modelo e primeira versao do main.py)
