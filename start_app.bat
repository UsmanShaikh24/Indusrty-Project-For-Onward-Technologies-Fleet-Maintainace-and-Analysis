@echo off
echo 🚛 FLEET MAINTENANCE PREDICTIVE ANALYTICS
echo ============================================================
echo Starting the app directly with Python (no Docker, no API needed!)
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found!
    echo Please install Python 3.9+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\fleet_maintenance_model.pkl" (
    echo ❌ Error: Trained models not found!
    echo Please run 'python enhanced_ml_engine.py' first to train the models.
    pause
    exit /b 1
)

echo ✅ All components ready!
echo.
echo 🚀 Launching Streamlit Dashboard...
echo 📱 The app will open in your default web browser
echo 🔗 URL: http://localhost:8501
echo.
echo 💡 What you get:
echo    ✅ Complete ML predictions (no API needed!)
echo    ✅ Interactive dashboard with real-time analytics
echo    ✅ File upload for batch processing
echo    ✅ All features working locally
echo.
echo ⏹️  Press Ctrl+C to stop the app
echo ============================================================

REM Launch Streamlit app
python -m streamlit run fleet_maintenance_app.py --server.port 8501

pause
