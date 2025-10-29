@echo off
REM Script to run the Log Analyzer Flask app in virtual environment

echo Activating virtual environment...
call env\Scripts\activate.bat

echo.
echo Starting Log Analyzer Web UI...
echo Open your browser and go to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python ui\app.py

pause
