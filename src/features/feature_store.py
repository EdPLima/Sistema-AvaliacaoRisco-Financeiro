import pickle
import pandas as pd
from src.utils.paths import data_path

class FeatureStore:
    """
    Feature Store para INFERÊNCIA.
    Responsável por aplicar o mesmo preprocessing e seleção de features
    usados no treinamento.

    - 1 carregar artefatos do pré-processamento
    - 2 carregar selecao final de features
    - 3 aplicar t
    """
    # Construtor da classe - inicializa os atributos
    def __init__(self):
        self.preprocessor = None # Preprocessor completo (imputacao + scaling + encoding)
        self.selected_features = None  # features selecionadas
        self._loaded = False           # garantir que a store foi carregada

    @classmethod
    def load(cls) -> "FeatureStore":
        store = cls()

        # Define o diretório dos artefatos de preprocessamento
        scalers_dir = data_path("", "scalers")

        # Carrega o pipeline completo
        with open(scalers_dir / "preprocessor.pkl", "rb") as f:
            store.preprocessor = pickle.load(f)

        # Carrega as variáveis selecionadas
        with open(scalers_dir / "feature_selection.pkl", "rb") as f:
            store.selected_features = pickle.load(f)["selected_features"]

        # Marca a store como carregada
        store._loaded = True
        return store


    # Recebe dados brutos do usuário
    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        # Verifica se a store foi carredada
        if not self._loaded:
            raise RuntimeError("FeatureStore não carregada. Use FeatureStore.load().")

        # Preprocessamento completo
        X_full = self.preprocessor.transform(df_raw)

        X_full = pd.DataFrame(
            X_full,
            columns=self.preprocessor.get_feature_names_out(),
            index=df_raw.index
        )

        # Seleção final de features
        return X_full[self.selected_features]