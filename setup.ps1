$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { throw "Python 3.11+ is required. Install Python from python.org and enable the PATH option." }
$VersionText = & python --version
if ($VersionText -notmatch "Python 3\.(1[1-9]|[2-9][0-9])") { throw "Python 3.11 or later is required; found $VersionText." }
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
$Venv = Join-Path $ProjectRoot ".venv"
if (-not (Test-Path -LiteralPath (Join-Path $Venv "Scripts\python.exe"))) {
    & python -m venv $Venv
}
& (Join-Path $Venv "Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $Venv "Scripts\python.exe") -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".env"))) {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot ".env.example") -Destination (Join-Path $ProjectRoot ".env")
}
$Ollama = Get-Command ollama -ErrorAction SilentlyContinue
if ($Ollama) {
    Write-Host "Ollama executable found:"
    & ollama --version
    try { & ollama list } catch { Write-Host "Ollama is installed but its local service is not available." }
} else {
    Write-Host "Ollama is not installed. Template mode works without it."
}
Write-Host ""
Write-Host "Setup complete. Edit .env if you want Ollama or Groq, then run .\run.ps1"
