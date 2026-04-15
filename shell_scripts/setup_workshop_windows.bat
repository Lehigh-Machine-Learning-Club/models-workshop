@echo off
setlocal enabledelayedexpansion

REM Workshop setup script for Windows (cmd/.bat)
REM Run from the shell_scripts folder:
REM   setup_workshop_windows.bat
REM Optional flags:
REM   --recompute-toy
REM   --recompute-mnist

set RECOMPUTE_TOY=0
set RECOMPUTE_MNIST=0

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--recompute-toy" (
  set RECOMPUTE_TOY=1
  shift
  goto parse_args
)
if /I "%~1"=="--recompute-mnist" (
  set RECOMPUTE_MNIST=1
  shift
  goto parse_args
)
echo Unknown option: %~1
echo Usage: setup_workshop_windows.bat [--recompute-toy] [--recompute-mnist]
exit /b 1

:args_done
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%\.."
if errorlevel 1 (
  echo ERROR: Could not locate project root.
  exit /b 1
)
set PROJECT_ROOT=%CD%
echo ==> Project root: %PROJECT_ROOT%

set PYTHON_BIN=%PYTHON_BIN%
if "%PYTHON_BIN%"=="" set PYTHON_BIN=python

where "%PYTHON_BIN%" >nul 2>nul
if errorlevel 1 (
  echo ERROR: Python not found. Install Python 3.9+ and retry.
  popd
  exit /b 1
)

echo ==> Creating virtual environment (.venv)
"%PYTHON_BIN%" -m venv .venv
if errorlevel 1 (
  echo ERROR: Failed to create virtual environment.
  popd
  exit /b 1
)

set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_ROOT%\.venv\Scripts\pip.exe

echo ==> Installing/refreshing dependencies
"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: Failed to upgrade pip.
  popd
  exit /b 1
)
"%VENV_PIP%" install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Failed to install requirements.
  popd
  exit /b 1
)

set TOY_FILE=%PROJECT_ROOT%\models\toy_checkpoints_sigmoid.npz
set MNIST_FILE=%PROJECT_ROOT%\models\mnist_mlp.pt

if "%RECOMPUTE_TOY%"=="1" (
  echo ==> Recomputing toy checkpoints
  "%VENV_PYTHON%" scripts\precompute_toy_training.py
  if errorlevel 1 (
    echo ERROR: Toy checkpoint generation failed.
    popd
    exit /b 1
  )
) else (
  if exist "%TOY_FILE%" (
    echo ==> Toy checkpoints found, skipping recompute
  ) else (
    echo ==> Toy checkpoints missing, generating now
    "%VENV_PYTHON%" scripts\precompute_toy_training.py
    if errorlevel 1 (
      echo ERROR: Toy checkpoint generation failed.
      popd
      exit /b 1
    )
  )
)

if "%RECOMPUTE_MNIST%"=="1" (
  echo ==> Recomputing MNIST artifacts
  "%VENV_PYTHON%" scripts\train_mnist.py
  if errorlevel 1 (
    echo ERROR: MNIST training failed.
    popd
    exit /b 1
  )
) else (
  if exist "%MNIST_FILE%" (
    echo ==> MNIST model artifact found, skipping retrain
  ) else (
    echo ==> MNIST model artifact missing, training now
    "%VENV_PYTHON%" scripts\train_mnist.py
    if errorlevel 1 (
      echo ERROR: MNIST training failed.
      popd
      exit /b 1
    )
  )
)

echo ==> Launching Streamlit app
"%VENV_PYTHON%" -m streamlit run app.py

popd
exit /b 0
