@echo off
echo ============================================
echo Medical Imaging Diagnostic System - Setup
echo ============================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
echo.

echo [2/3] Installing Frontend dependencies...
cd frontend
call npm install
cd ..
echo.

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Next Steps:
echo 1. Train models: python backend/train.py --dataset aptos --epochs 20
echo 2. Start backend: python backend/main.py
echo 3. Start frontend: cd frontend ; npm run dev
echo.
echo Optional: Set GOOGLE_API_KEY for enhanced chatbot
echo   $env:GOOGLE_API_KEY="your-key"
echo.
pause
