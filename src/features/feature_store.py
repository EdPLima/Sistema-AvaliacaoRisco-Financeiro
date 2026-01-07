import pickle
import pandas as pd
import logging
from src.utils.paths import data_path

# Sistema de registro de eventos
logger = logging.getLogger(__name__)

class FeatureStore:
    """
    Camada responsável por padronizar o acesso às features em producao

    - Consistência entre treino e inferencia do modelo
    - Aplicacao do pipeline de inferencia a mesma que foi feita na etapa de modelagem
    - Selecao correta das features finanis utilizadas pelo modelo

    """

    # Construtor da classe
    def __init__(self):
        self.preprocessor = None        # Pipeline de pré-processamento treinado
        self.selected_features = None   # Lista final de features usadas no modelo
        self._loaded = False            # Controle de carregamento da store

    @classmethod
    # retorna um objeto FeatureStore
    def load(cls) -> "FeatureStore":

        """Carrega os artefatos necessários para inferência."""

        store = cls()
        scalers_dir = data_path("", "scalers")

        #-----------------------------------------
        #Carregamento do preprocessor
        #-----------------------------------------

        try:
            # Observabilidade
            logger.info(f"Carregando preprocessor de: {scalers_dir / 'preprocessor.pkl'}")    # logger.info fluxo normal do sistema

            with open(scalers_dir / "preprocessor.pkl", "rb") as f:
                store.preprocessor = pickle.load(f)

            logger.info("Preprocessor carregado com sucesso")

        except FileNotFoundError as e:
            # Falha crítica quando o arquivo .pkl não é encontrado
            logger.error(f"Arquivo preprocessor.pkl não encontrado: {e}")
            raise

        except Exception as e:
            # Falha Crítica erros de compatibilidade ou versao do artefato
            logger.error(f"Erro ao carregar preprocessor: {e}")
            raise


        #-------------------------------------------
        # Carregamento da seleção de features
        # ------------------------------------------

        try:
            logger.info(f"Carregando feature_selection de: {scalers_dir / 'feature_selection.pkl'}")

            with open(scalers_dir / "feature_selection.pkl", "rb") as f:
                feature_data = pickle.load(f)
                store.selected_features = feature_data["selected_features"]

            logger.info(f"Feature selection carregado: {len(store.selected_features)} features selecionadas")

            # Observabilidade: amostra dos artefato ( Nao impacta selecao real)
            logger.debug(f"Primeiras 5 features: {store.selected_features[:5]}")

        except FileNotFoundError as e:
            # Falha crítica: modelo perde contrato de features
            logger.error(f"Arquivo feature_selection.pkl não encontrado: {e}")
            raise

        except KeyError as e:
            # Indica inconsistência de schema do artefato salvo
            logger.error(f"Chave 'selected_features' não encontrada no arquivo: {e}")
            raise

        except Exception as e:
            logger.error(f"Erro ao carregar feature_selection: {e}")
            raise

        # Marca explicitamente que a FeatureStore está pronta para uso
        store._loaded = True
        return store


    def transform_all(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica o pipeline completo de pré-processamento aos dados brutos
        Retorna todas as features transformadas, sem aplicar seleção.
        """

        if not self._loaded:
            # Proteção contra uso incorreto do componente
            raise RuntimeError("FeatureStore não carregada. Use FeatureStore.load().")

        try:
            # Observabilidade: shape de entrata para detectar quebra de contrato
            logger.info(f"Transformando DataFrame com shape: {df_raw.shape}")
            logger.debug(f"Colunas de entrada: {list(df_raw.columns)}")

            # Aplicação do pipeline treinado
            X_full = self.preprocessor.transform(df_raw)
            feature_names = self.preprocessor.get_feature_names_out()

            logger.info(f"Transformação concluída. Shape: {X_full.shape}")

            # Observabilidade inspeção das features geradas
            logger.debug(f"Primeiras 10 colunas transformadas: {list(feature_names)[:10]}")

            # Reconstrói Dataframe preservando índice original
            X_full = pd.DataFrame(
                X_full,
                columns=feature_names,
                index=df_raw.index
            )

            return X_full

        except Exception as e:
            # erro indica mismatch entre chema de entrada e o pipeline treinado
            logger.error(f"Erro ao transformar dados: {e}")
            logger.error(f"Colunas esperadas pelo preprocessor podem não corresponder às fornecidas")
            raise

    def select_features(self, X_full: pd.DataFrame) -> pd.DataFrame:
        """Seleciona as features finais utilizadas pelo Modelo

        -  nomes de features salvos no treino
        - nomes expandidos após pré-processamento

        """
        all_columns = list(X_full.columns)
        updated_selected_features = []

        logger.info(f"Selecionando features. Total de colunas disponíveis: {len(all_columns)}")
        logger.info(f"Total de features a selecionar: {len(self.selected_features)}")

        for f in self.selected_features:
            # Tenta múltiplas estratégias de matching:

            # Match exato
            if f in all_columns:
                updated_selected_features.append(f)
                continue

            # Match com endswith para features transformadas como "num__person_income"
            matches = [c for c in all_columns if c.endswith(f"__{f}") or c.endswith(f)]

            # Match com contains
            if not matches:
                matches = [c for c in all_columns if f in c]

            if matches:
                updated_selected_features.extend(matches)

                # Observabilidade: mapeamento explícito para debug
                logger.debug(f"Feature '{f}' mapeada para: {matches}")
            else:

                # Warning pois o modelo ainda pode funcionar
                logger.warning(f"Feature '{f}' não encontrada nas colunas transformadas!")

        if not updated_selected_features:
            # Falha crítica: nenhuma feature do modelo foi encontrada
            error_msg = (
                f"Nenhuma feature do selected_features bateu com as colunas transformadas!\n"
                f"Features esperadas (primeiras 10): {self.selected_features[:10]}\n"
                f"Colunas disponíveis (primeiras 10): {all_columns[:10]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Remove duplicatas mantendo ordem
        # Evita colunas repetidas em cenários de múltiplos matches
        updated_selected_features = list(dict.fromkeys(updated_selected_features))

        logger.info(f"Features selecionadas: {len(updated_selected_features)}")

        # Observabilidade: amostra das features finais usadas na inferência
        logger.debug(f"Primeiras 10 features selecionadas: {updated_selected_features[:10]}")

        return X_full[updated_selected_features]


    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Interface única para inferência
        Encapsula o fluxo completo:

        - pré-processamento
        - selecao de features

        Este deve ser o método padrão utilizado por serviços de previsão.
        """
        X_full = self.transform_all(df_raw)
        X_selected = self.select_features(X_full)
        return X_selected
