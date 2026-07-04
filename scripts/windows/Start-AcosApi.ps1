param(
  [string]$AcosDir = "C:\Users\jalan\wip\acos",
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

Set-Location $AcosDir
New-Item -ItemType Directory -Force -Path ".acos\logs", ".acos\jobs-ui", ".acos\ui-cycles" | Out-Null

$env:LOCAL_ORNITH_BASE_URL = "http://127.0.0.1:8000/v1"
$env:LOCAL_ORNITH_TIMEOUT_SECONDS = "1200"
$env:ORNITH_API_KEY = ""

$python = Join-Path $AcosDir ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  $python = "python"
}

& $python -m uvicorn apps.api.main:create_app --factory --host $HostAddress --port $Port
