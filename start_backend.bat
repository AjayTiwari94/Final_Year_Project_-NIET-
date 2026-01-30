@echo off
echo ============================================
echo Starting Backend Server...
echo ============================================
echo.
cd /d "%~dp0"
python backend/main.py
pause
