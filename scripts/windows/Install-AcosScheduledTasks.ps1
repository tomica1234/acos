param(
  [string]$AcosDir = "C:\Users\jalan\wip\acos",
  [string]$FrontendDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($FrontendDir)) {
  $FrontendDir = Join-Path $AcosDir "frontend"
}

$apiScript = Join-Path $AcosDir "scripts\windows\Start-AcosApi.ps1"
$frontendScript = Join-Path $AcosDir "scripts\windows\Start-AcosFrontend.ps1"
$powerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"

$apiAction = New-ScheduledTaskAction `
  -Execute $powerShell `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$apiScript`" -AcosDir `"$AcosDir`""
$frontendAction = New-ScheduledTaskAction `
  -Execute $powerShell `
  -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$frontendScript`" -FrontendDir `"$FrontendDir`""

$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet `
  -AllowStartIfOnBatteries `
  -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Days 365) `
  -RestartCount 999 `
  -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
  -TaskName "ACOS API" `
  -Action $apiAction `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Runs the ACOS API on Windows for remote progress and background jobs." `
  -Force | Out-Null

Register-ScheduledTask `
  -TaskName "ACOS Frontend" `
  -Action $frontendAction `
  -Trigger $trigger `
  -Settings $settings `
  -Description "Runs the ACOS frontend on Windows." `
  -Force | Out-Null

Start-ScheduledTask -TaskName "ACOS API"
Start-ScheduledTask -TaskName "ACOS Frontend"

Write-Host "Installed and started ACOS scheduled tasks."
Write-Host "API:      http://127.0.0.1:8080"
Write-Host "Frontend: http://127.0.0.1:5174"
