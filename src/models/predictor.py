import pandas as pd
import numpy as np
from src.models.loader_model import load_production_model

class ModelProducao:

    def __init__(self, model_name: str = "lgb_prob_default"):
        self.model_name = model_name
        self._modelo = load_production_model(model_name)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna as probabilidades por classe.
        Returns:
            np.ndarray shape (n_samples, 2)
        """
        return self._modelo.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Retorna a classe prevista (0 ou 1)
        """
        return self._modelo.predict(X)
