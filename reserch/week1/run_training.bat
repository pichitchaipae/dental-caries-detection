@echo off
REM ============================================================
REM Dental X-ray Segmentation - Training Setup Script
REM Optimized for: RTX 4080 (12GB) + Ryzen 9 7945HX
REM ============================================================

echo ============================================================
echo Dental X-ray Tooth Segmentation - YOLO11-seg Training
echo Hardware: RTX 4080 (12GB) + Ryzen 9 7945HX
echo ============================================================
echo.

REM Activate conda environment
call conda activate dental
if errorlevel 1 (
    echo ERROR: Failed to activate 'dental' conda environment.
    echo Please run: conda create -n dental python=3.10 -y
    echo Then: conda activate dental
    echo Then: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    echo Then: pip install ultralytics pillow
    pause
    exit /b 1
)

echo Environment: dental
echo.

REM Verify CUDA setup
echo Verifying CUDA setup...
python verify_cuda.py
if errorlevel 1 (
    echo.
    echo ERROR: CUDA verification failed!
    echo Please fix the issues above before training.
    pause
    exit /b 1
)

echo.

REM Check if dataset exists
if not exist "dataset\images\train" (
    echo Dataset not found. Running preparation script...
    python prepare_dataset.py
    if errorlevel 1 (
        echo ERROR: Dataset preparation failed.
        pause
        exit /b 1
    )
)

echo.
echo Starting training on RTX 4080...
echo ============================================================
python train.py

echo.
echo ============================================================
echo Training complete!
echo Results saved in: runs\
echo ============================================================
pause
