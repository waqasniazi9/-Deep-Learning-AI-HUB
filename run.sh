#!/bin/bash
# Quick Start Script for AI Learning Hub

echo "🚀 AI & Deep Learning Hub - Quick Start"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix-like
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

echo "✅ Dependencies installed"

# Run the app
echo ""
echo "🌐 Starting AI Learning Hub..."
echo "📍 Open your browser at: http://localhost:8501"
echo ""

streamlit run online_app.py
