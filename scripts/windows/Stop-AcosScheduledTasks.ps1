$ErrorActionPreference = "SilentlyContinue"

Stop-ScheduledTask -TaskName "ACOS API"
Stop-ScheduledTask -TaskName "ACOS Frontend"
Unregister-ScheduledTask -TaskName "ACOS API" -Confirm:$false
Unregister-ScheduledTask -TaskName "ACOS Frontend" -Confirm:$false

Write-Host "Stopped and removed ACOS scheduled tasks."
