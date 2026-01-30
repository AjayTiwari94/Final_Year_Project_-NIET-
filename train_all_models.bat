@echo off
echo ============================================
echo Training All Medical Imaging Models
echo ============================================
echo.
echo This will train 3 models sequentially:
echo 1. APTOS (Retinal Scan)
echo 2. HAM10000 (Skin Lesion)
echo 3. MURA (X-Ray)
echo.
echo Estimated time: 6-12 hours (GPU) or 24+ hours (CPU)
echo.
pause
echo.

cd /d "%~dp0"

echo ============================================
echo [1/3] Training APTOS Model...
echo ============================================
python backend/train.py --dataset aptos --epochs 20
echo.

echo ============================================
echo [2/3] Training HAM10000 Model...
echo ============================================
python backend/train.py --dataset ham10000 --epochs 20
echo.

echo ============================================
echo [3/3] Training MURA Model...
echo ============================================
python backend/train.py --dataset mura --epochs 20
echo.

echo ============================================
echo All Models Trained Successfully!
echo ============================================
echo.
echo Models saved in: backend/saved_models/
echo.
echo You can now run the application:
echo 1. Double-click start_backend.bat
echo 2. Double-click start_frontend.bat
echo.
pause
