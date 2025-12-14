from pathlib import Path

# Caminho absoluto para a raiz do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

# Pastas principais
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"
MODELS_DIR = BASE_DIR / "src" / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
MLRUNS_DIR = BASE_DIR / "mlruns"

def ensure_folder(path: Path):
    """Cria a pasta se não existir"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def data_path(filename: str, subfolder: str = "") -> Path:
    """Retorna caminho completo para arquivos na pasta data/ (opcional subpasta)"""
    folder = ensure_folder(DATA_DIR / subfolder)
    return folder / filename

def config_path(filename: str, subfolder: str = "") -> Path:
    """Retorna caminho completo para arquivos na pasta configs/ (opcional subpasta)"""
    folder = ensure_folder(CONFIGS_DIR / subfolder)
    return folder / filename

def model_path(filename: str, subfolder: str = "") -> Path:
    """Retorna caminho completo para arquivos na pasta models/ (opcional subpasta)"""
    folder = ensure_folder(MODELS_DIR / subfolder)
    return folder / filename

def experiments_path(filename: str, subfolder: str = "") -> Path:
    """Retorna caminho completo para arquivos na pasta experiments/ (opcional subpasta)"""
    folder = ensure_folder(EXPERIMENTS_DIR / subfolder)
    return folder / filename

def mlruns_path(filename: str, subfolder: str = "") -> Path:
    """Retorna caminho completo para arquivos na pasta mlruns/ (opcional subpasta)"""
    folder = ensure_folder(MLRUNS_DIR / subfolder)
    return folder / filename

