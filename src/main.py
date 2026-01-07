import pandas as pd
import numpy as np
from typing import Union, Dict
from src.models.predictor import ModelProducao
from src.features.feature_store import FeatureStore

def prever_risco(dados_entrada: Union[Dict, pd.DataFrame], threshold: float = 0.42) -> Dict:
    # Converter entrada para DataFrame
    if isinstance(dados_entrada, dict):
        df_input = pd.DataFrame([dados_entrada])
    else:
        df_input = dados_entrada.copy()

    # Carregar FeatureStore
    feature_store = FeatureStore.load()

    # Passo 1: transforma todas as features
    X_full = feature_store.transform_all(df_input)

    # Passo 2: seleciona apenas as features do RFECV
    X_final = feature_store.select_features(X_full)

    # Carregar modelo de produção
    modelo = ModelProducao()
    proba = modelo.predict_proba(X_final)

    # Probabilidade da classe 1 (inadimplência)
    prob_default = proba[0, 1]

    # Classificação
    classificacao = "Alto Risco" if prob_default >= threshold else "Baixo Risco"
    confianca = abs(prob_default - threshold)

    return {
        'probabilidade_default': round(prob_default, 4),
        'classificacao': classificacao,
        'confianca': round(confianca, 4),
        'threshold_usado': threshold
    }


def prever_risco_lote(dados_lote: pd.DataFrame, threshold: float = 0.42) -> pd.DataFrame:
    df_input = dados_lote.copy()

    # Carregar FeatureStore
    feature_store = FeatureStore.load()

    # Passo 1: transforma todas as features
    X_full = feature_store.transform_all(df_input)

    # Passo 2: seleciona apenas as features do RFECV
    X_final = feature_store.select_features(X_full)

    # Carregar modelo de produção
    modelo = ModelProducao()
    proba = modelo.predict_proba(X_final)
    prob_default = proba[:, 1]

    # Classificação
    classificacao = np.where(prob_default >= threshold, "Alto Risco", "Baixo Risco")
    confianca = np.abs(prob_default - threshold)

    resultado = pd.DataFrame({
        'probabilidade_default': np.round(prob_default, 4),
        'classificacao': classificacao,
        'confianca': np.round(confianca, 4)
    })

    return resultado
