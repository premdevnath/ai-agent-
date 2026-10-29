@echo off
setlocal
cd /d "%~dp0"
title Financial RAG Chatbot - Setup and Run
echo.
echo ==========================================
echo   Financial RAG Chatbot - Windows Launcher
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating a Python virtual environment...
    where py >nul 2>nul
    if errorlevel 1 (
        echo ERROR: Python launcher was not found.
        echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/windows/
        echo During installation, tick "Add python.exe to PATH".
        pause
        exit /b 1
    )
    py -3.12 -m venv .venv >nul 2>nul
    if errorlevel 1 py -3.11 -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create the virtual environment.
        echo Please install Python 3.11 or 3.12, then run this file again.
        pause
        exit /b 1
    )
)

echo [2/4] Checking/installing required Python packages.
echo This can take a few minutes the first time; keep the window open.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip could not be updated. Check your internet connection.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Dependency installation failed.
    echo Check your internet connection and the error shown above.
    pause
    exit /b 1
)

echo [3/4] Starting the app...
echo [4/4] When Streamlit opens, enter your OpenAI API key in the left sidebar.
echo Do not share your API key or paste it into GitHub.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The app stopped with an error. Copy the error text and ask for help.
    pause
)
endlocal
