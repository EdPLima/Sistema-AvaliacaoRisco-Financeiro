import pandas as pd
import numpy as np
from src.models.loader_model import load_production_model

class ModelProducao:
    """
    Camada de abstração para inferência do modelo em produção.

    Responsabilidades:
    - Carregamento do modelo versionado
    - Padronização de chamadas de predição
    - Redução de acoplamento com a implementação interna
    """

    def __init__(self, model_name: str = "lgb_prob_default"):
        self.model_name = model_name
        # Carrega o modelo pronto para inferência
        self._modelo = load_production_model(model_name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades por classe (0 e 1).
        Saída: array de shape (n_samples, 2)
        """
        # O modelo LightGBM carregado diretamente tem predict_proba()
        if not hasattr(self._modelo, 'predict_proba'):
            raise AttributeError(
                f"Modelo do tipo {type(self._modelo)} não possui método predict_proba(). "
                f"O modelo deve ser um LightGBM Classifier."
            )

        proba = self._modelo.predict_proba(X)

        # Garantir que está no formato correto (n_samples, 2)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
            proba = np.column_stack([1 - proba, proba])
        elif proba.shape[1] != 2:
            raise ValueError(
                f"Formato de saída inesperado do modelo: shape {proba.shape}. "
                f"Esperado: (n_samples, 2)"
            )

        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna a classe prevista (0 ou 1).
        Saída compatível com pipelines downstream.
        """
        return self._modelo.predict(X)
