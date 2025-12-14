import mlflow
from typing import Optional

def carrega_modelo(model_name: str = 'lgb_prob_default', tracking_uri: Optional[str] = None):
    """
    Carrega o modelo promovido para Production no MLflow.
    Args:
        model_name: Nome do modelo no MLflow Registry
        tracking_uri: URI do MLflow (se None, detecta automaticamente)
    """
    # Configura MLflow
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        from src.utils.paths import experiments_path
        tracking_dir = experiments_path("").resolve()
        mlflow.set_tracking_uri(tracking_dir.as_uri())

    # Carrega modelo de Production
    model_uri = f"models:/{model_name}/Production"
    modelo = mlflow.sklearn.load_model(model_uri)
    return modelo
