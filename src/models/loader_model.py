import mlflow
import pandas as pd
import numpy as np

def load_production_model(model_name: str):
    """
    Carrega o modelo promovido para Production no MLflow.
    Args:
        model_name: Nome do modelo no MLflow Registry
    """
    # Carrega modelo de Production diretamente
    model_uri = f"models:/{model_name}/Production"
    modelo = mlflow.sklearn.load_model(model_uri)
    return modelo
