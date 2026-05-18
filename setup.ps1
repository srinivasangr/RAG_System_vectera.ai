# Convenience setup for Windows PowerShell.
# Same four steps as setup.sh: venv, deps, .env check, Snowflake bootstrap.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$App  = Join-Path $Root "app"

Write-Host "─── 1/4  Python venv ──────────────────────────────────────────" -ForegroundColor Cyan
$Venv = Join-Path $App ".venv"
if (-not (Test-Path $Venv)) {
    python -m venv $Venv
    Write-Host "   created $Venv"
} else {
    Write-Host "   already exists"
}
$Activate = Join-Path $Venv "Scripts\Activate.ps1"
. $Activate

Write-Host ""
Write-Host "─── 2/4  install requirements ─────────────────────────────────" -ForegroundColor Cyan
& pip install --quiet --upgrade pip
& pip install --quiet -r (Join-Path $App "requirements.txt")

Write-Host ""
Write-Host "─── 3/4  check .env ───────────────────────────────────────────" -ForegroundColor Cyan
$EnvFile = Join-Path $App ".env"
if (-not (Test-Path $EnvFile)) {
    Copy-Item (Join-Path $App ".env.example") $EnvFile
    Write-Host "   created a fresh app/.env from the example; fill in your keys and re-run."
    exit 1
}

$Required = @("SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "CEREBRAS_API_KEY")
$Missing = @()
$EnvLines = Get-Content $EnvFile
foreach ($k in $Required) {
    $line = $EnvLines | Where-Object { $_ -match "^${k}=" } | Select-Object -First 1
    $val = if ($line) { ($line -split "=", 2)[1].Trim('"').Trim() } else { "" }
    if (-not $val -or $val -like "*YOUR_*" -or $val -like "*PASTE_*") {
        $Missing += $k
    }
}
if ($Missing.Count -gt 0) {
    Write-Host "   the following required keys are empty or placeholder in app/.env:" -ForegroundColor Yellow
    $Missing | ForEach-Object { Write-Host "     - $_" }
    exit 1
}
Write-Host "   all required keys present"

Write-Host ""
Write-Host "─── 4/4  Snowflake schema ─────────────────────────────────────" -ForegroundColor Cyan
Set-Location $App
python -m rag_system.storage.init_snowflake

Write-Host ""
Write-Host "✓ setup complete" -ForegroundColor Green
Write-Host ""
Write-Host "Next:"
Write-Host "   cd app; .\.venv\Scripts\Activate.ps1"
Write-Host "   python scripts/ingest_all_pending.py --no-vision    # ingest the 10 PDFs"
Write-Host "   streamlit run rag_system/ui/streamlit_app.py        # open the chat UI"
