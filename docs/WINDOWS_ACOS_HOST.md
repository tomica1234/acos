# Windows ACOS Host

This setup runs ACOS on the Windows machine that already hosts Ornith.

## Target Layout

- ACOS repo: `C:\Users\jalan\wip\acos`
- ACOS frontend: `C:\Users\jalan\wip\acos\frontend`
- Ornith lazy proxy: `http://127.0.0.1:8000/v1`
- ACOS API: `http://0.0.0.0:8080`
- ACOS frontend: `http://0.0.0.0:5174`

From the Mac over Tailscale, open:

```text
http://100.95.69.79:5174
```

## Install On Windows

Run these in PowerShell:

```powershell
cd C:\Users\jalan\wip\acos
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install pytest

cd C:\Users\jalan\wip\acos\frontend
npm install

cd C:\Users\jalan\wip\acos
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Install-AcosScheduledTasks.ps1
```

## Verify

```powershell
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:5174
```

Expected Ornith model:

```text
ornith-1.0-35b-Q4_K_M
```

## Stop

```powershell
cd C:\Users\jalan\wip\acos
powershell -ExecutionPolicy Bypass -File .\scripts\windows\Stop-AcosScheduledTasks.ps1
```

## Notes

The Mac can be closed because it is no longer the ACOS host. The Windows machine must stay powered on, logged in enough for user scheduled tasks, and reachable over Tailscale.
