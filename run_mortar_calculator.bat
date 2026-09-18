@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>&1
    if errorlevel 1 (
        echo Python 3 is required. Install it from https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo Creating the Python environment...
    py -m venv .venv
    if errorlevel 1 (
        echo Could not create the Python environment.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import cv2, mss, pynput, rapidocr_onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo Installing the calculator packages...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Package installation failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" mortar_calculator.py %*
if errorlevel 1 pause
