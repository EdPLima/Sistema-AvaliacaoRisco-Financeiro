from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import traceback
import logging
import sys

from src.features.feature_store import FeatureStore
from src.models.predictor import ModelProducao

# Configurar logging com mais detalhes
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SingleInput(BaseModel):
    features: Dict[str, Any]
    threshold: Optional[float] = Field(0.42, ge=0.0, le=1.0)
    
    class Config:
        # Permitir que features seja um dicionário aninhado ou plano
        extra = "allow"


class BatchInput(BaseModel):
    records: List[Dict[str, Any]]
    threshold: Optional[float] = Field(0.42, ge=0.0, le=1.0)


app = FastAPI(title="Credit Risk Prediction API")


# Handler global de exceções
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura todas as exceções não tratadas e retorna uma resposta JSON detalhada"""
    error_detail = str(exc)
    error_traceback = traceback.format_exc()
    
    logger.error(f"Erro não tratado na requisição {request.url.path}: {error_detail}")
    logger.error(f"Traceback completo:\n{error_traceback}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": error_detail,
            "traceback": error_traceback,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handler específico para erros de validação do Pydantic"""
    logger.error(f"Erro de validação na requisição {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "Dados de entrada inválidos",
            "details": exc.errors()
        }
    )


@app.get("/health")
def health():
    """Endpoint de health check básico"""
    return {"status": "ok"}


@app.get("/test-model")
def test_model():
    """Endpoint de teste para verificar se o modelo e FeatureStore podem ser carregados"""
    results = {
        "status": "testing",
        "feature_store": {},
        "model": {}
    }
    
    try:
        # Testar FeatureStore
        logger.info("Testando carregamento do FeatureStore...")
        feature_store = FeatureStore.load()
        results["feature_store"] = {
            "loaded": True,
            "preprocessor_loaded": feature_store.preprocessor is not None,
            "selected_features_count": len(feature_store.selected_features) if feature_store.selected_features else 0
        }
        logger.info("✓ FeatureStore carregado com sucesso")
    except Exception as e:
        results["feature_store"] = {
            "loaded": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"✗ Erro ao carregar FeatureStore: {e}")
    
    try:
        # Testar Modelo
        logger.info("Testando carregamento do modelo...")
        modelo = ModelProducao()
        results["model"] = {
            "loaded": True,
            "model_name": modelo.model_name
        }
        logger.info("✓ Modelo carregado com sucesso")
    except Exception as e:
        results["model"] = {
            "loaded": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        logger.error(f"✗ Erro ao carregar modelo: {e}")
    
    if results["feature_store"].get("loaded") and results["model"].get("loaded"):
        results["status"] = "ok"
    else:
        results["status"] = "error"
    
    return results


@app.get("/debug/check-files")
def check_files():
    """Endpoint de debug para verificar se os arquivos necessários existem"""
    from pathlib import Path
    from src.utils.paths import data_path
    
    results = {
        "status": "checking",
        "files": {},
        "paths": {}
    }

    try:
        scalers_dir = data_path("", "scalers")
        results["paths"]["scalers_dir"] = str(scalers_dir)
        results["paths"]["scalers_dir_exists"] = scalers_dir.exists()

        preprocessor_path = scalers_dir / "preprocessor.pkl"
        feature_selection_path = scalers_dir / "feature_selection.pkl"

        results["files"]["preprocessor.pkl"] = {
            "path": str(preprocessor_path),
            "exists": preprocessor_path.exists(),
            "size": preprocessor_path.stat().st_size if preprocessor_path.exists() else 0
        }

        results["files"]["feature_selection.pkl"] = {
            "path": str(feature_selection_path),
            "exists": feature_selection_path.exists(),
            "size": feature_selection_path.stat().st_size if feature_selection_path.exists() else 0
        }

        # Tentar carregar FeatureStore
        try:
            feature_store = FeatureStore.load()
            results["feature_store"] = {
                "loaded": True,
                "preprocessor_loaded": feature_store.preprocessor is not None,
                "selected_features_count": len(feature_store.selected_features) if feature_store.selected_features else 0
            }
        except Exception as e:
            results["feature_store"] = {
                "loaded": False,
                "error": str(e)
            }

        # Verificar MLflow
        import os
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "not set")
        results["mlflow"] = {
            "tracking_uri": mlflow_uri,
            "tracking_uri_set": mlflow_uri != "not set"
        }

        results["status"] = "ok"

    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

    return results


@app.post("/predict")
def predict(payload: SingleInput):
    logger.info("=" * 80)
    logger.info("Nova requisição de predição recebida")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Recebendo predição com features: {list(payload.features.keys())}")
        logger.info(f"Tipo de payload.features: {type(payload.features)}")
        logger.info(f"Conteúdo de payload.features: {payload.features}")
        logger.info(f"Threshold recebido: {payload.threshold}")
        
        # Garantir que payload.features é um dicionário e criar DataFrame corretamente
        if not isinstance(payload.features, dict):
            raise ValueError(f"features deve ser um dicionário, recebido: {type(payload.features)}")
        
        # Se features contém uma chave 'features' (payload aninhado), desaninhar
        features_dict = payload.features
        if 'features' in features_dict and isinstance(features_dict['features'], dict):
            logger.warning("Payload aninhado detectado, desaninhando...")
            features_dict = features_dict['features']
        
        # Criar DataFrame a partir do dicionário de features
        df = pd.DataFrame([features_dict])
        logger.info(f"DataFrame criado com shape: {df.shape}, colunas: {list(df.columns)}")
        
        # Verificar se as colunas esperadas estão presentes
        expected_cols = [
            'person_income', 'person_home_ownership', 'person_emp_length',
            'loan_intent', 'loan_grade', 'loan_amnt', 'loan_int_rate',
            'loan_percent_income', 'cb_person_default_on_file',
            'cb_person_cred_hist_length', 'faixa_etaria'
        ]
        missing_cols = set(expected_cols) - set(df.columns)
        if missing_cols:
            error_msg = (
                f"Colunas faltando: {missing_cols}\n"
                f"Colunas recebidas: {list(df.columns)}\n"
                f"Features recebidas: {features_dict}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as exc:
        logger.error(f"Erro ao criar DataFrame: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"invalid input: {exc}")

    try:
        # Carregar FeatureStore e aplicar pipeline completo
        logger.info("=" * 80)
        logger.info("Iniciando carregamento do FeatureStore...")
        logger.info("=" * 80)
        
        # Verificar caminhos antes de carregar
        from src.utils.paths import data_path
        scalers_dir = data_path("", "scalers")
        logger.info(f"Caminho do diretório scalers: {scalers_dir}")
        logger.info(f"Diretório scalers existe: {scalers_dir.exists()}")
        
        if scalers_dir.exists():
            preprocessor_path = scalers_dir / "preprocessor.pkl"
            feature_selection_path = scalers_dir / "feature_selection.pkl"
            logger.info(f"preprocessor.pkl existe: {preprocessor_path.exists()}")
            logger.info(f"feature_selection.pkl existe: {feature_selection_path.exists()}")
        
        feature_store = FeatureStore.load()
        logger.info("✓ FeatureStore carregado com sucesso")

        logger.info("Aplicando transform_all...")
        X_full = feature_store.transform_all(df)
        logger.info(f"✓ X_full shape: {X_full.shape}, colunas: {list(X_full.columns)[:10]}...")

        logger.info("Aplicando select_features...")
        X_final = feature_store.select_features(X_full)
        logger.info(f"✓ X_final shape: {X_final.shape}, colunas: {list(X_final.columns)[:10]}...")
    except Exception as exc:
        error_detail = str(exc)
        error_traceback = traceback.format_exc()
        logger.error("=" * 80)
        logger.error(f"ERRO NO FEATURESTORE: {error_detail}")
        logger.error("=" * 80)
        logger.error(f"Traceback completo:\n{error_traceback}")
        # Retornar mais informações no erro para debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "feature store error",
                "message": error_detail,
                "traceback": error_traceback
            }
        )

    try:
        logger.info("=" * 80)
        logger.info("Iniciando carregamento do modelo...")
        logger.info("=" * 80)
        
        # Verificar variável de ambiente MLflow
        import os
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "not set")
        logger.info(f"MLFLOW_TRACKING_URI: {mlflow_uri}")
        
        modelo = ModelProducao()
        logger.info("✓ Modelo carregado com sucesso")
        
        logger.info("Fazendo predição...")
        logger.info(f"Shape do X_final para predição: {X_final.shape}")
        proba = modelo.predict_proba(X_final)
        logger.info(f"Shape do resultado da predição: {proba.shape}")
        prob_default = float(proba[0, 1])
        logger.info(f"✓ Probabilidade calculada: {prob_default}")
    except Exception as exc:
        error_detail = str(exc)
        error_traceback = traceback.format_exc()
        logger.error("=" * 80)
        logger.error(f"ERRO NA INFERÊNCIA DO MODELO: {error_detail}")
        logger.error("=" * 80)
        logger.error(f"Traceback completo:\n{error_traceback}")
        # Retornar mais informações no erro para debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "model inference error",
                "message": error_detail,
                "traceback": error_traceback
            }
        )

    threshold = float(payload.threshold)
    classificacao = "Alto Risco" if prob_default >= threshold else "Baixo Risco"
    confianca = abs(prob_default - threshold)
    
    # Calcula nível de risco baseado em faixas de probabilidade
    # Baixo: <= 30%, Médio: <= 60%, Alto: > 60%
    if prob_default <= 0.30:
        nivel_risco = "Baixo"
    elif prob_default <= 0.60:
        nivel_risco = "Médio"
    else:
        nivel_risco = "Alto"
    
    # Calcula nível de confiança normalizado (0.0 a 1.0)
    nivel_confianca = min(confianca * 2, 1.0)

    return {
        "probabilidade_default": round(prob_default, 4),
        "probabilidade_percentual": round(prob_default * 100, 2),
        "classificacao": classificacao,
        "nivel_risco": nivel_risco,
        "confianca": round(confianca, 4),
        "nivel_confianca": round(nivel_confianca, 4),
        "threshold_usado": threshold,
    }


@app.post("/predict_batch")
def predict_batch(payload: BatchInput):
    try:
        logger.info(f"Recebendo predição em lote com {len(payload.records)} registros")
        df = pd.DataFrame(payload.records)
        logger.info(f"DataFrame criado com shape: {df.shape}")
    except Exception as exc:
        logger.error(f"Erro ao criar DataFrame: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"invalid input: {exc}")

    try:
        # Carregar FeatureStore e aplicar pipeline completo
        logger.info("Carregando FeatureStore...")
        feature_store = FeatureStore.load()
        logger.info("Aplicando transformações...")
        X_full = feature_store.transform_all(df)
        X_final = feature_store.select_features(X_full)
        logger.info(f"Features transformadas: {X_final.shape}")
    except Exception as exc:
        error_detail = str(exc)
        error_traceback = traceback.format_exc()
        logger.error(f"Erro no FeatureStore: {error_detail}\n{error_traceback}")
        # Retornar mais informações no erro para debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "feature store error",
                "message": error_detail,
                "traceback": error_traceback
            }
        )

    try:
        logger.info("Carregando modelo e fazendo predições...")
        modelo = ModelProducao()
        proba = modelo.predict_proba(X_final)
        prob_default = proba[:, 1].astype(float)
        logger.info(f"Predições concluídas para {len(prob_default)} registros")
    except Exception as exc:
        error_detail = str(exc)
        error_traceback = traceback.format_exc()
        logger.error(f"Erro na inferência do modelo: {error_detail}\n{error_traceback}")
        # Retornar mais informações no erro para debug
        return JSONResponse(
            status_code=500,
            content={
                "error": "model inference error",
                "message": error_detail,
                "traceback": error_traceback
            }
        )

    threshold = float(payload.threshold)
    results = []
    for p in prob_default:
        classificacao = "Alto Risco" if p >= threshold else "Baixo Risco"
        confianca = abs(p - threshold)
        results.append({
            "probabilidade_default": round(float(p), 4),
            "classificacao": classificacao,
            "confianca": round(float(confianca), 4),
        })

    return {"results": results, "threshold_usado": threshold}
