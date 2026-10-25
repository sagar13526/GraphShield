@echo off
setlocal
cd /d "%~dp0"
title GraphShield Training Pipeline

echo ==========================================
echo GraphShield - Synthetic Fraud Detection
echo ==========================================
python --version
if errorlevel 1 (
  echo Python was not found. Install Python and enable it in PATH.
  pause
  exit /b 1
)

echo.
echo Checking/installing project dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed. Copy the error above for troubleshooting.
  pause
  exit /b 1
)

echo.
echo Running the complete pipeline...
python src\main.py
if errorlevel 1 (
  echo The project stopped with an error. Copy the final error lines for troubleshooting.
  pause
  exit /b 1
)
echo.
echo Success. Open the results folder or run run_dashboard.bat.
pause
