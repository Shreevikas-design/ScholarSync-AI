$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run .\setup.ps1 first."
}
$env:TEMP = Join-Path $ProjectRoot ".tmp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $ProjectRoot ".pip-cache"
$env:PYTHONPYCACHEPREFIX = Join-Path $ProjectRoot ".pycache"
$env:STREAMLIT_CONFIG_DIR = Join-Path $ProjectRoot ".streamlit"
$env:HF_HOME = Join-Path $ProjectRoot "models\\huggingface"
$env:HUGGINGFACE_HUB_CACHE = Join-Path $ProjectRoot "models\\huggingface\\hub"
$env:SENTENCE_TRANSFORMERS_HOME = Join-Path $ProjectRoot "models\\sentence-transformers"
$env:OLLAMA_MODELS = Join-Path $ProjectRoot "models\\ollama"
New-Item -ItemType Directory -Force -Path $env:TEMP, $env:PIP_CACHE_DIR, $env:PYTHONPYCACHEPREFIX, $env:STREAMLIT_CONFIG_DIR, $env:HF_HOME, $env:HUGGINGFACE_HUB_CACHE, $env:SENTENCE_TRANSFORMERS_HOME, $env:OLLAMA_MODELS | Out-Null
. (Join-Path $Venv "Scripts\Activate.ps1")
Set-Location -LiteralPath $ProjectRoot
& (Join-Path $Venv "Scripts\streamlit.exe") run app.py --server.address 127.0.0.1
