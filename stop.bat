@echo off
:: ============================================================
::  GUI-X  |  Glass Matrix OS
::  Orchestration Script — Stop
:: ============================================================
setlocal enabledelayedexpansion

title GUI-X — Stopping

echo.
echo  ============================================================
echo   GUI-X  ^|  Glass Matrix OS  —  Stop
echo  ============================================================
echo.

set PID_FILE=%~dp0.server.pid

:: ── Stop via saved PID ───────────────────────────────────────
if exist "%PID_FILE%" (
    set /p SERVER_PID=<"%PID_FILE%"
    echo  [INFO]  Stopping process PID !SERVER_PID! ...
    taskkill /PID !SERVER_PID! /F >nul 2>&1
    if !errorlevel! == 0 (
        echo  [OK]   Process !SERVER_PID! terminated.
    ) else (
        echo  [WARN]  Could not kill PID !SERVER_PID! — may have already exited.
    )
    del /f /q "%PID_FILE%" >nul 2>&1
) else (
    echo  [INFO]  No .server.pid file found.
)

:: ── Also kill any lingering PowerShell hosting server.ps1 ────
echo  [INFO]  Scanning for lingering GUI-X server processes...
where pwsh >nul 2>&1
if %errorlevel% == 0 (
    for /f "tokens=2" %%P in ('tasklist /fi "imagename eq pwsh.exe" /fo csv /nh 2^>nul ^| findstr /i "pwsh"') do (
        set RAW_PID=%%P
        set CLEAN_PID=!RAW_PID:"=!
        taskkill /PID !CLEAN_PID! /F >nul 2>&1
        echo  [INFO]  Killed pwsh.exe PID !CLEAN_PID!
    )
)

for /f "tokens=2" %%P in ('tasklist /fi "imagename eq powershell.exe" /fo csv /nh 2^>nul ^| findstr /i "powershell"') do (
    set RAW_PID=%%P
    set CLEAN_PID=!RAW_PID:"=!
    taskkill /PID !CLEAN_PID! /F >nul 2>&1
    echo  [INFO]  Killed powershell.exe PID !CLEAN_PID!
)

:: ── Kill the console window titled "GUI-X Server" ────────────
taskkill /fi "windowtitle eq GUI-X Server" /F >nul 2>&1

echo.
echo  [DONE]  GUI-X server stopped.
echo.
endlocal
