import pickle
import pandas as pd
from src.utils.paths import data_path


class FeatureStore:
    """
    Responsável por:
    - Carregar artefatos de pré-processamento
    - Aplicar transformações idênticas ao treino
    - Garantir consistência de features em produção
    """

    def __init__(self):
        self.preprocessor = None        # Pipeline de pré-processamento treinado
        self.selected_features = None   # Lista final de features usadas no modelo
        self._loaded = False            # Controle de carregamento da store

    @classmethod
    def load(cls) -> "FeatureStore":
        """Carrega os artefatos necessários para inferência."""
        store = cls()
        scalers_dir = data_path("", "scalers")

        with open(scalers_dir / "preprocessor.pkl", "rb") as f:
            store.preprocessor = pickle.load(f)

        with open(scalers_dir / "feature_selection.pkl", "rb") as f:
            store.selected_features = pickle.load(f)["selected_features"]

        store._loaded = True
        return store

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Aplica pré-processamento e seleção final de features."""
        if not self._loaded:
            raise RuntimeError("FeatureStore não carregada. Use FeatureStore.load().")

        X_full = self.preprocessor.transform(df_raw)

        X_full = pd.DataFrame(
            X_full,
            columns=self.preprocessor.get_feature_names_out(),
            index=df_raw.index
        )

        return X_full[self.selected_features]
