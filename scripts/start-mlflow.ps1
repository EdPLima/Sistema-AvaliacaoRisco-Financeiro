# Script para iniciar o MLflow UI apontando automaticamente para a pasta `experiments/mlruns`
# Execute no PowerShell a partir da raiz do repositório:
#   .\scripts\start-mlflow.ps1

param(
    [int]$Port = 5000,
    [string]$Host = '127.0.0.1'
)

# Resolve repo root (assumes script is in scripts/ under repo root)
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
$mlrunsPath = Join-Path $repoRoot 'experiments\mlruns'

if (-not (Test-Path $mlrunsPath)) {
    Write-Error "Diretório de tracking não encontrado: $mlrunsPath"
    exit 1
}

# Convert Windows path to file:/// URI (use forward slashes)
$abs = (Resolve-Path $mlrunsPath).Path
$uri = 'file:///' + ($abs -replace '\\','/')

Write-Host "Starting MLflow UI using backend-store-uri: $uri"

mlflow ui --backend-store-uri $uri --default-artifact-root $uri --host $Host --port $Port
