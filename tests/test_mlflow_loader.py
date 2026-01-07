"""
Testes unitários para o carregador de modelos do MLflow.
"""
import pytest
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml

# Adicionar o diretório raiz ao path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.loader_model import load_production_model


class TestMLflowLoader:
    """Testes para o carregador de modelos do MLflow."""

    @pytest.fixture
    def mock_mlflow_tracking_uri(self):
        """Mock do tracking URI do MLflow."""
        with patch('src.models.loader_model.mlflow.set_tracking_uri') as mock_set:
            yield mock_set

    @pytest.fixture
    def mock_alias_file(self, tmp_path):
        """Cria um arquivo de alias 'Production' para versao do modelo"""
        alias_dir = tmp_path / "models" / "lgb_prob_default_production" / "aliases"
        alias_dir.mkdir(parents=True)
        alias_file = alias_dir / "Production"
        alias_file.write_text("4")
        return alias_file

    @pytest.fixture
    def mock_meta_file(self, tmp_path):
        """Cria um arquivo meta.yaml mock."""
        version_dir = tmp_path / "models" / "lgb_prob_default_production" / "version-4"
        version_dir.mkdir(parents=True)
        meta_file = version_dir / "meta.yaml"
        meta_content = {
            'run_id': '6867b6fc502d468d95038db97f292687',
            'model_id': 'm-08fbf774f4344b748c297a11d50ae7fb',
            'version': '4'
        }
        meta_file.write_text(yaml.dump(meta_content))
        return meta_file

    def test_load_model_with_alias(self, mock_mlflow_tracking_uri, tmp_path, monkeypatch):
        """Testa o carregamento do modelo usando alias Production."""
        mlruns_path = tmp_path / "experiments" / "mlruns"
        mlruns_path.mkdir(parents=True)

        experiment_id = "585793176030616435"
        model_id = "m-08fbf774f4344b748c297a11d50ae7fb"
        model_name = "lgb_prob_default_production"

        # Criar estrutura
        (mlruns_path / "models" / model_name / "aliases" / "Production").parent.mkdir(parents=True)
        (mlruns_path / "models" / model_name / "aliases" / "Production").write_text("4")

        (mlruns_path / "models" / model_name / "version-4").mkdir(parents=True)
        meta_file = mlruns_path / "models" / model_name / "version-4" / "meta.yaml"
        meta_file.write_text(yaml.dump({'model_id': model_id}))

        (mlruns_path / experiment_id / "models" / model_id / "artifacts").mkdir(parents=True)
        (mlruns_path / experiment_id / "models" / model_id / "artifacts" / "MLmodel").write_text("model_type: lightgbm")

        mock_model = MagicMock()

        # Mock Path para retornar caminhos do tmp_path quando chamado com /app/experiments/mlruns
        original_path = Path
        def mock_path(*args):
            if args and str(args[0]) == "/app/experiments/mlruns/models":
                return mlruns_path / "models"
            elif args and str(args[0]) == "/app/experiments/mlruns":
                return mlruns_path
            return original_path(*args)

        with patch('src.models.loader_model.Path', side_effect=mock_path):
            with patch('src.models.loader_model.mlflow.pyfunc.load_model', return_value=mock_model):
                model = load_production_model(model_name)
                assert model is not None
                assert model == mock_model

    def test_load_model_fallback_to_latest_version(self, mock_mlflow_tracking_uri, tmp_path):
        """Testa o fallback para versão mais recente quando alias não existe."""
        # Esta é uma estrutura mais simples para testar o fallback
        pass  # Implementar se necessário

    def test_load_model_handles_missing_files(self, mock_mlflow_tracking_uri):
        """Testa o comportamento quando arquivos necessários não existem."""
        # Mock para simular arquivos ausentes
        with patch('src.models.loader_model.Path') as mock_path:
            mock_path.return_value.exists.return_value = False

            with pytest.raises(Exception):
                load_production_model("modelo_inexistente")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

