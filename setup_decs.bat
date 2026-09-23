@echo off
title DECS Smart Setup
color 0a
cd /d "%~dp0"

echo ===================================================
echo     DECS ROBUST & SMART ENVIRONMENT SETUP
echo ===================================================
echo [*] Project folder: %CD%
echo.

:: --- Check for namespace collision issues ---
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

:: --- Locate Base Python Interpreter ---
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
set "VENV_DIR=%CD%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

:: --- Create Virtual Environment if Missing ---
if not exist "%VENV_PY%" (
    echo [*] Virtual environment not found. Creating new .venv ...
    %BASEPY% -m venv .venv
    if not exist "%VENV_PY%" (
        echo [ERROR] Failed to create virtual environment at %VENV_PY%
        pause
        exit /b 1
    )
) else (
    echo [*] Existing virtual environment found.
)

:: --- Check and Repair Pip inside .venv ---
echo [*] Verifying pip inside virtual environment...
"%VENV_PY%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [!] Pip missing or corrupt in .venv. Bootstrapping pip via ensurepip...
    "%VENV_PY%" -m ensurepip --upgrade >nul 2>&1
    "%VENV_PY%" -m pip --version >nul 2>&1
    if errorlevel 1 (
        echo [!] ensurepip failed. Re-creating clean virtual environment...
        rmdir /s /q "%VENV_DIR%"
        %BASEPY% -m venv .venv
    )
)

:: --- Check Health of Installed Dependencies ---
echo [*] Checking health of core dependencies...
"%VENV_PY%" -c "import cv2, fastapi, uvicorn, ultralytics, websockets, torch" >nul 2>&1
if not errorlevel 1 (
    echo [*] Core dependencies are intact! Skipping unnecessary reinstallations.
    goto :verify
)

echo [*] Dependency mismatch or missing packages found. Updating environment...

:: --- Upgrade pip safely ---
echo.
echo [*] Ensuring pip is up to date...
"%VENV_PY%" -m pip install --upgrade pip >nul 2>&1

:: --- Selective dependency installation ---
echo.
echo [*] Installing/Updating required project dependencies...
"%VENV_PY%" -m pip install -r requirements.txt --upgrade-strategy only-if-needed
if errorlevel 1 (
    echo [ERROR] pip install failed. Scroll up to inspect the error details.
    pause
    exit /b 1
)

:verify
echo.
echo [*] Verifying critical imports inside the virtual environment...
"%VENV_PY%" -c "import cv2; print('cv2 OK ->', cv2.__version__)"
if errorlevel 1 (
    echo [ERROR] cv2 import failed inside the venv.
    pause
    exit /b 1
)
"%VENV_PY%" -c "import fastapi, uvicorn, ultralytics, websockets, torch; print('Core dependencies OK')"
if errorlevel 1 (
    echo [ERROR] A core dependency failed to import inside the venv.
    pause
    exit /b 1
)

echo.
if not exist "mediamtx.exe" (
    echo [WARNING] mediamtx.exe is missing from this folder.
    echo           Download it from https://github.com/bluenviron/mediamtx/releases
    echo           and place mediamtx.exe here before running run_decs.bat
) else (
    echo [OK] mediamtx.exe found.
)

echo.
echo ===================================================
echo [SUCCESS] Setup complete. Environment is ready.
echo Run run_decs.bat to start the tactical system.
echo ===================================================
pause