@echo off
setlocal enabledelayedexpansion
title DECS C2 Tactical Master Launcher
color 0b

:: 1. Force the execution context to the project root directory
cd /d "%~dp0"

echo ================================================================
echo         TACTICAL DECS C2 MASTER AUTOMATION LAUNCHER
echo ================================================================
echo [*] System Root: %CD%

:: 2. Locate MediaMTX executable
set "MEDIAMTX_CMD="
if exist "%CD%\mediamtx.exe" (
    set "MEDIAMTX_CMD=%CD%\mediamtx.exe"
) else (
    for /f "delims=" %%I in ('where mediamtx.exe 2^>nul') do set "MEDIAMTX_CMD=%%I"
)

if not defined MEDIAMTX_CMD (
    echo [!] WARNING: mediamtx.exe not found in %CD% or PATH.
    echo     Ensure MediaMTX is started manually if hosted elsewhere.
) else (
    echo [*] Node 1/3: Launching MediaMTX RTMP Server...
    start "DECS - MediaMTX Server" /d "%CD%" cmd /k ""!MEDIAMTX_CMD!""
)

:: Give MediaMTX 2 seconds to bind port 1935
timeout /t 2 /nobreak >nul

:: 3. Launch FastAPI Telemetry Backend via Python module resolution
echo [*] Node 2/3: Launching FastAPI Telemetry Hub (:8000)...
start "DECS - Telemetry Backend" /d "%CD%" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

:: Give Uvicorn 3 seconds to bind port 8000 and initialize models
timeout /t 3 /nobreak >nul

:: 4. Launch Ingestion & Detection Engine
echo [*] Node 3/3: Launching RTMP Detection Engine...
start "DECS - Detection Engine" /d "%CD%" cmd /k "python .\drone_ingestion\rtmp_reader.py --input_size 416"

:: 5. Open Web Dashboard
timeout /t 1 /nobreak >nul
echo [*] Launching C2 Dashboard in browser...
start http://localhost:8000

echo.
echo ================================================================
echo [SUCCESS] All nodes spawned successfully.
echo Keep the individual node windows open while running.
echo ================================================================
pause