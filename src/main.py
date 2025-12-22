import pandas as pd
import numpy as np
from typing import Union, Dict, Tuple
from src.models.loader_model import ModeloProducao
from features.feature_store import FeatureStore


def prever_risco(dados_entrada: Union[Dict, pd.DataFrame],threshold: float = 0.42) -> Dict:

    if isinstance(dados_entrada, dict):
        df_input = pd.DataFrame([dados_entrada])
    else:
        df_input = dados_entrada.copy()

    # Carregar e aplicar transformações de features
    feature_store = FeatureStore.load()
    X_transformado = feature_store.transform(df_input)

    # Carregar modelo de produção que tá no mlflow
    modelo = ModeloProducao()

    # Fazer predição de probabilidades
    proba = modelo.predict_proba(X_transformado)

    # Probabilidade da classe 1 que é a classe que o cara nao vai pagar
    prob_default = proba[0, 1]

    # Classificar em Alto/Baixo Risco com base no threshold
    classificacao = "Alto Risco" if prob_default >= threshold else "Baixo Risco"

    # Calcular confiança - distância em relação ao threshold
    confianca = abs(prob_default - threshold)

    return {
        'probabilidade_default': round(prob_default, 4),
        'classificacao': classificacao,
        'confianca': round(confianca, 4),
        'threshold_usado': threshold
    }

def prever_risco_lote(dados_lote: pd.DataFrame,threshold: float = 0.42) -> pd.DataFrame:

    # Transformar features
    feature_store = FeatureStore.load()
    X_transformado = feature_store.transform(dados_lote)

    # Carregar modelo
    modelo = ModeloProducao()

    # Predição em lote
    proba = modelo.predict_proba(X_transformado)
    prob_default = proba[:, 1]

    # Classificar
    classificacao = np.where(prob_default >= threshold, "Alto Risco", "Baixo Risco")
    confianca = np.abs(prob_default - threshold)

    resultado = pd.DataFrame({
        'probabilidade_default': np.round(prob_default, 4),
        'classificacao': classificacao,
        'confianca': np.round(confianca, 4)
    })

    return resultado
