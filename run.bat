@echo off
echo ========================================
echo Classroom Safety Narrator - Quick Start
echo ========================================
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Warning: Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo Starting Flask server...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause

