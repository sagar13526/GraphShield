@echo off
setlocal
cd /d "%~dp0"
title GraphShield Dashboard

echo ==========================================
echo GraphShield - Fraud Analytics Dashboard
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

if not exist "data\accounts.csv" goto train_first
if not exist "data\transactions.csv" goto train_first
if not exist "data\graph.npz" goto train_first
if not exist "results\metrics.json" goto train_first
if not exist "results\test_predictions.npz" goto train_first
goto launch

:train_first
echo Required data/results are missing. Running the full pipeline first...
python src\main.py
if errorlevel 1 (
  echo The training pipeline failed. Copy the final error lines for troubleshooting.
  pause
  exit /b 1
)

:launch
echo.
echo Starting dashboard. Keep this window open while using it.
python -m streamlit run app.py
if errorlevel 1 (
  echo The dashboard stopped with an error. Copy the final error lines for troubleshooting.
  pause
  exit /b 1
)
pause
