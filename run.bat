@echo off
REM Quick Start Script for AI Learning Hub (Windows)

echo 🚀 AI Deep Learning Hub - Quick Start
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

echo ✅ Python found

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated

REM Install dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

echo ✅ Dependencies installed

REM Run the app
echo.
echo 🌐 Starting AI Learning Hub...
echo 📍 Open your browser at: http://localhost:8501
echo.

streamlit run online_app.py

pause
