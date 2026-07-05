param(
  [string]$FrontendDir = "C:\Users\jalan\wip\acos\frontend",
  [string]$HostAddress = "0.0.0.0",
  [int]$Port = 5174
)

$ErrorActionPreference = "Stop"

Set-Location $FrontendDir

$portableNode = "C:\Users\jalan\wip\tools\node-v22.13.0-win-x64"
if ($env:ACOS_NODE_HOME -and (Test-Path (Join-Path $env:ACOS_NODE_HOME "node.exe"))) {
  $env:PATH = "$env:ACOS_NODE_HOME;$env:PATH"
} elseif (Test-Path (Join-Path $portableNode "node.exe")) {
  $env:PATH = "$portableNode;$env:PATH"
} elseif (Test-Path "C:\Program Files\nodejs\node.exe") {
  $env:PATH = "C:\Program Files\nodejs;$env:PATH"
}

$npm = (Get-Command npm.cmd -ErrorAction Stop).Source
& $npm run dev -- --host $HostAddress --port $Port
