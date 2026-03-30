@echo off
:: ============================================================
::  GUI-X  |  Glass Matrix OS
::  Orchestration Script — Start
:: ============================================================
setlocal enabledelayedexpansion

title GUI-X Launcher

echo.
echo  ============================================================
echo   GUI-X  ^|  Glass Matrix OS  —  Launcher
echo  ============================================================
echo.

:: ── Check PowerShell availability ───────────────────────────
where pwsh >nul 2>&1
if %errorlevel% == 0 (
    set PS_EXE=pwsh
) else (
    where powershell >nul 2>&1
    if %errorlevel% == 0 (
        set PS_EXE=powershell
    ) else (
        echo  [ERROR] PowerShell not found. Please install PowerShell.
        echo  https://github.com/PowerShell/PowerShell/releases
        pause
        exit /b 1
    )
)

echo  [INFO]  Using: %PS_EXE%

:: ── Check if already running ────────────────────────────────
if exist "%~dp0.server.pid" (
    set /p EXISTING_PID=<"%~dp0.server.pid"
    echo  [WARN]  Server may already be running (PID !EXISTING_PID!)
    echo  [WARN]  Run stop.bat first, or press any key to continue anyway.
    pause
)

:: ── Resolve script directory ─────────────────────────────────
set SCRIPT_DIR=%~dp0
:: Remove trailing backslash
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo  [INFO]  Root directory: %SCRIPT_DIR%
echo  [INFO]  Starting server on http://localhost:3000 ...
echo.

:: ── Launch PowerShell server in a new window ─────────────────
start "GUI-X Server" %PS_EXE% -NoProfile -ExecutionPolicy Bypass ^
    -File "%SCRIPT_DIR%\server.ps1" ^
    -Port 3000 ^
    -Root "%SCRIPT_DIR%"

:: ── Wait for server to be ready (poll /health) ───────────────
set RETRIES=0
:WAIT_LOOP
if %RETRIES% geq 15 (
    echo  [WARN]  Server did not respond in time — opening browser anyway.
    goto OPEN_BROWSER
)
timeout /t 1 /nobreak >nul
%PS_EXE% -NoProfile -Command ^
    "try { $r=(Invoke-WebRequest 'http://localhost:3000/health' -UseBasicParsing -TimeoutSec 2).StatusCode; exit ($r -ne 200) } catch { exit 1 }" >nul 2>&1
if %errorlevel% == 0 goto SERVER_READY
set /a RETRIES+=1
echo  [....] Waiting for server (%RETRIES%/15)...
goto WAIT_LOOP

:SERVER_READY
echo  [OK]   Server is up!

:OPEN_BROWSER
:: ── Open default browser ────────────────────────────────────
echo  [INFO]  Opening http://localhost:3000 in your browser...
start "" "http://localhost:3000"

echo.
echo  ============================================================
echo   GUI-X is running at http://localhost:3000
echo   Close the "GUI-X Server" window or run stop.bat to quit.
echo  ============================================================
echo.
endlocal
