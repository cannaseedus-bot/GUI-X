@echo off
:: ============================================================
::  GUI-X  |  Glass Matrix OS
::  Orchestration Script — Install / Setup Check
:: ============================================================
setlocal enabledelayedexpansion

title GUI-X — Install

echo.
echo  ============================================================
echo   GUI-X  ^|  Glass Matrix OS  —  Setup Check
echo  ============================================================
echo.

set ERRORS=0

:: ── 1. Check PowerShell ──────────────────────────────────────
echo  [CHECK] PowerShell...
where pwsh >nul 2>&1
if %errorlevel% == 0 (
    for /f "delims=" %%V in ('pwsh -NoProfile -Command "$PSVersionTable.PSVersion.ToString()" 2^>nul') do set PS_VER=%%V
    echo  [OK]    PowerShell 7+  v!PS_VER!  (pwsh)
    set PS_EXE=pwsh
) else (
    where powershell >nul 2>&1
    if %errorlevel% == 0 (
        for /f "delims=" %%V in ('powershell -NoProfile -Command "$PSVersionTable.PSVersion.ToString()" 2^>nul') do set PS_VER=%%V
        echo  [OK]    Windows PowerShell  v!PS_VER!  (powershell)
        echo  [NOTE]  For best performance, install PowerShell 7+:
        echo          https://github.com/PowerShell/PowerShell/releases
        set PS_EXE=powershell
    ) else (
        echo  [FAIL]  PowerShell NOT found!
        set /a ERRORS+=1
    )
)

:: ── 2. Check index.html exists ───────────────────────────────
echo  [CHECK] index.html...
if exist "%~dp0index.html" (
    echo  [OK]    index.html found.
) else (
    echo  [FAIL]  index.html not found in %~dp0
    set /a ERRORS+=1
)

:: ── 3. Check server.ps1 exists ───────────────────────────────
echo  [CHECK] server.ps1...
if exist "%~dp0server.ps1" (
    echo  [OK]    server.ps1 found.
) else (
    echo  [FAIL]  server.ps1 not found in %~dp0
    set /a ERRORS+=1
)

:: ── 4. Check port 3000 availability ─────────────────────────
echo  [CHECK] Port 3000...
%PS_EXE% -NoProfile -Command ^
    "try { $l=[System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback,3000); $l.Start(); $l.Stop(); exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel% == 0 (
    echo  [OK]    Port 3000 is available.
) else (
    echo  [WARN]  Port 3000 may already be in use.
    echo         Run stop.bat or change the port in server.ps1 / start.bat.
)

:: ── 5. Set PowerShell execution policy (user scope) ──────────
echo  [CHECK] Execution policy...
%PS_EXE% -NoProfile -Command ^
    "Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force" >nul 2>&1
if %errorlevel% == 0 (
    echo  [OK]    Execution policy set to RemoteSigned (CurrentUser).
) else (
    echo  [WARN]  Could not set execution policy — run as Administrator if needed.
)

:: ── Summary ──────────────────────────────────────────────────
echo.
if !ERRORS! == 0 (
    echo  ============================================================
    echo   All checks passed! Run start.bat to launch GUI-X.
    echo  ============================================================
) else (
    echo  ============================================================
    echo   !ERRORS! error(s) found. Please resolve them before running.
    echo  ============================================================
)
echo.
pause
endlocal
