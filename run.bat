@echo off
echo ========================================
echo   SUSTAINED GPU MAXIMIZER
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "sustained_gpu_maximizer.py" (
    echo ERROR: sustained_gpu_maximizer.py not found
    echo Please run this script from the project directory
    pause
    exit /b 1
)

echo Checking setup...
python test_setup.py

echo.
echo ========================================
echo Choose an option:
echo 1. Run example with sample data
echo 2. Process your CSV files
echo 3. Test setup only
echo 4. Exit
echo ========================================
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Running example...
    python run_example.py
) else if "%choice%"=="2" (
    echo.
    echo Processing your CSV files...
    python sustained_gpu_maximizer.py
) else if "%choice%"=="3" (
    echo.
    echo Setup test completed above.
) else if "%choice%"=="4" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please run the script again.
)

echo.
pause