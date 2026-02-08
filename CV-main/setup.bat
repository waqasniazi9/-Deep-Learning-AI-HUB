@echo off
echo 🚀 Deep Learning Notebook Hub - Setup ^& Launch
echo ================================================

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo ✅ Setup complete!
echo.
echo Choose how to run:
echo 1. Streamlit Dashboard (Recommended): streamlit run app.py
echo 2. Voila Web Apps: voila notebook_name.ipynb
echo 3. Jupyter Notebook: jupyter notebook
echo 4. Docker Compose: docker-compose up
echo.
pause
