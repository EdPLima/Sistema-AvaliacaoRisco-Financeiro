# Script para iniciar a API (FastAPI) com `uvicorn`.
# Execute no PowerShell a partir da raiz do repositório:
#   .\scripts\start-api.ps1

param(
    [int]$Port = 8000,
    [string]$Host = '127.0.0.1'
)

Write-Host "Starting API on http://$Host`:$Port"
python -m uvicorn src.api.predict:app --host $Host --port $Port
