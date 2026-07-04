param(
  [string]$FrontendDir = "C:\Users\jalan\wip\acos\frontend",
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 5174
)

$ErrorActionPreference = "Stop"

Set-Location $FrontendDir
$env:PATH = "C:\Program Files\nodejs;$env:PATH"

npm run dev -- --host $HostAddress --port $Port
