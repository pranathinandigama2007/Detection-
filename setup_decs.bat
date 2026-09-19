@echo off
title DECS Setup
color 0a
cd /d "%~dp0"

echo ===================================================
echo     DECS ONE-TIME ENVIRONMENT SETUP
echo ===================================================
echo [*] Project folder: %CD%
echo.

if exist "cv2" (
    echo [ERROR] A folder named "cv2" exists in this project directory.
    echo         This shadows the real opencv-python package and breaks every import.
    echo         Rename or delete it, then re-run this script.
    pause
    exit /b 1
)
if exist "cv2.py" (
    echo [ERROR] A file named "cv2.py" exists in this project directory.
    echo         This shadows the real opencv-python package and breaks every import.
    echo         Rename or delete it, then re-run this script.
    pause
    exit /b 1
)

echo [*] Locating a Python interpreter...
set "BASEPY="
py -3.11 -c "print(1)" >nul 2>&1
if not errorlevel 1 (
    set "BASEPY=py -3.11"
    goto :found
)
py -3 -c "print(1)" >nul 2>&1
if not errorlevel 1 (
    set "BASEPY=py -3"
    goto :found
)
python -c "print(1)" >nul 2>&1
if not errorlevel 1 (
    set "BASEPY=python"
    goto :found
)

echo [ERROR] No working Python interpreter found via "py -3.11", "py -3", or "python".
echo         Install Python 3.11 from https://www.python.org/downloads/ and
echo         make sure to check "Add python.exe to PATH" during install.
pause
exit /b 1

:found
echo [*] Using base interpreter: %BASEPY%
%BASEPY% --version

echo.
echo [*] Creating isolated virtual environment in ".venv" ...
if exist ".venv" (
    echo [*] ".venv" already exists. Deleting it for a clean rebuild...
    rmdir /s /q ".venv"
)
%BASEPY% -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment creation failed. ".venv\Scripts\python.exe" was not created.
    pause
    exit /b 1
)

set "VENV_PY=%CD%\.venv\Scripts\python.exe"
echo [*] Virtual environment python: %VENV_PY%

echo.
echo [*] Upgrading pip inside the virtual environment...
"%VENV_PY%" -m pip install --upgrade pip

echo.
echo [*] Installing project dependencies (this downloads PyTorch/Ultralytics, can take several minutes)...
"%VENV_PY%" -m pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Scroll up to see which package failed and paste
    echo         the exact error back for a fix. Do not proceed to run_decs.bat yet.
    pause
    exit /b 1
)

echo.
echo [*] Verifying critical imports inside the virtual environment...
"%VENV_PY%" -c "import cv2; print('cv2 OK ->', cv2.__version__)"
if errorlevel 1 (
    echo [ERROR] cv2 import failed inside the fresh venv. Paste this exact error back.
    pause
    exit /b 1
)
"%VENV_PY%" -c "import fastapi, uvicorn, ultralytics, websockets, torch; print('core deps OK')"
if errorlevel 1 (
    echo [ERROR] A core dependency failed to import inside the fresh venv. Paste this exact error back.
    pause
    exit /b 1
)

echo.
if not exist "mediamtx.exe" (
    echo [WARNING] mediamtx.exe is still missing from this folder.
    echo           Download it from https://github.com/bluenviron/mediamtx/releases
    echo           (grab the Windows amd64 zip, extract it, copy mediamtx.exe here)
    echo           You can finish this later, but run_decs.bat will refuse to start without it.
) else (
    echo [OK] mediamtx.exe found.
)

echo.
echo ===================================================
echo [SUCCESS] Setup complete. Environment is ready.
echo Next step: place mediamtx.exe here if you haven't, then run run_decs.bat
echo ===================================================
pause
