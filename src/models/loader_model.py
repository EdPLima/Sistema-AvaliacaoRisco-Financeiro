import pickle
import pandas as pd
import numpy as np
import os
import logging
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

def load_production_model(model_name: str = "lgb_prob_default", version: int = None):
    """
    Carrega o modelo diretamente dos arquivos, sem usar MLflow.
    Args:
        model_name: Nome do modelo no MLflow Registry (padrão: lgb_prob_default)
        version: Versão do modelo. Se None, lê do alias Production (padrão: None)
    """
    logger.info("=" * 80)
    logger.info(f"Iniciando carregamento do modelo '{model_name}'")
    logger.info("=" * 80)

    # Caminho base dos experimentos
    base_path = Path("/app/experiments/mlruns")

    # Verificar se o caminho existe
    if not base_path.exists():
        raise FileNotFoundError(
            f"Diretório mlruns não encontrado em: {base_path}"
        )

    logger.info(f"Buscando modelo em: {base_path}")

    # Se a versão não foi especificada, ler do alias Production
    if version is None:
        alias_file = base_path / "models" / model_name / "aliases" / "Production"
        logger.info(f"Lendo versão do alias Production em: {alias_file}")

        if alias_file.exists():
            version = int(alias_file.read_text().strip())
            logger.info(f"Alias Production aponta para versão: {version}")
        else:
            # Fallback para versão 4 se o alias não existir
            logger.warning(f"Alias Production não encontrado, usando versão 4 como padrão")
            version = 4
    else:
        logger.info(f"Usando versão especificada: {version}")

    # Ler meta.yaml da versão para obter model_id
    version_meta_file = base_path / "models" / model_name / f"version-{version}" / "meta.yaml"
    logger.info(f"Verificando meta.yaml em: {version_meta_file}")

    if not version_meta_file.exists():
        raise FileNotFoundError(
            f"Meta.yaml não encontrado para versão {version}. "
            f"Caminho verificado: {version_meta_file}"
        )

    import yaml
    with open(version_meta_file, 'r') as f:
        meta = yaml.safe_load(f)

    model_id = meta.get('model_id')
    if not model_id:
        raise ValueError(f"model_id não encontrado no meta.yaml. Conteúdo: {meta}")

    logger.info(f"Model ID encontrado: {model_id}")

    # Buscar o modelo nos diretórios de experimentos
    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()]
    logger.info(f"Diretórios de experimentos encontrados: {[d.name for d in experiment_dirs]}")

    for exp_dir in experiment_dirs:
        model_path = exp_dir / "models" / model_id / "artifacts" / "model.pkl"
        logger.info(f"Verificando: {model_path}")

        if model_path.exists():
            logger.info(f"Arquivo do modelo encontrado em: {model_path}")
            try:
                # Carregar o modelo diretamente usando pickle
                with open(model_path, 'rb') as f:
                    modelo = pickle.load(f)

                logger.info(f"Modelo carregado com sucesso")
                logger.info(f"Tipo do modelo: {type(modelo)}")

                # Verificar se o modelo tem o método predict_proba
                if hasattr(modelo, 'predict_proba'):
                    logger.info("Modelo possui método predict_proba")
                else:
                    logger.warning("Modelo não possui método predict_proba")

                return modelo

            except Exception as e:
                logger.error(f"Erro ao carregar modelo de {model_path}: {e}")
                logger.error(traceback.format_exc())
                raise

    raise FileNotFoundError(
        f"Modelo não encontrado. Model ID: {model_id}, "
        f"Version: {version}, "
        f"Experiment dirs: {[d.name for d in experiment_dirs]}, "
        f"Base path: {base_path}"
    )
