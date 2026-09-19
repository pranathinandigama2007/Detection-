@echo off
title DECS C2 Tactical Launcher
color 0b
cd /d "%~dp0"

echo ===================================================
echo     LAUNCHING DECS C2 TACTICAL DRONE SYSTEM
echo ===================================================
echo [*] Working directory: %CD%

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at ".venv\Scripts\python.exe".
    echo         Run setup_decs.bat first, and let it finish successfully.
    pause
    exit /b 1
)
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

"%VENV_PY%" -c "import cv2, fastapi, uvicorn, ultralytics, torch" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] One or more core packages are not importable inside .venv.
    echo         Run setup_decs.bat again and check its output carefully.
    pause
    exit /b 1
)
echo [*] Using interpreter: %VENV_PY%

if not exist "mediamtx.exe" (
    echo [ERROR] mediamtx.exe not found in %CD%
    echo         Download it from https://github.com/bluenviron/mediamtx/releases
    echo         and place mediamtx.exe in this folder, then re-run this script.
    pause
    exit /b 1
)

echo [*] Node 1/2: Starting MediaMTX (RTMP:1935 -^> RTSP:8554 bridge)...
start "DECS - MediaMTX Server" /d "%CD%" cmd /k "mediamtx.exe"

timeout /t 2 /nobreak >nul

echo [*] Node 2/2: Starting FastAPI Tactical C2 Hub (:8000)...
start "DECS - Telemetry Backend" /d "%CD%" cmd /k ""%VENV_PY%" -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo [*] Waiting for the API to finish loading YOLO weights...
timeout /t 6 /nobreak >nul

echo [*] Opening C2 Dashboard in browser...
start http://localhost:8000

echo.
echo ===================================================
echo [DONE] Both DECS nodes launched. Keep both windows open.
echo.
echo NEXT STEPS ON THE DASHBOARD:
echo   1. Point your drone/phone RTMP app at: rtmp://THIS-PC-IP:1935/live/drone1
echo      (find THIS-PC-IP by running "ipconfig" in another terminal)
echo   2. On the dashboard, set Source Type = "RTMP DRONE/MOBILE BROADCAST"
echo   3. Leave the path as "live/drone1" (or match whatever path you streamed to)
echo   4. Click Connect. Video + detections should appear within a few seconds.
echo ===================================================
pause
