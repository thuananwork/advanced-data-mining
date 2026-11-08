@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

cd /d "%~dp0"

echo ============================================================================
echo   DU AN TABULAR FOUNDATION MODELS - CHAY THUC NGHIEM
echo   Thu muc: %CD%
echo ============================================================================

set "PY=.\.venv\Scripts\python.exe"

if exist "%PY%" goto :venv_ready
echo [1/2] Dang tao moi truong ao .venv...
python -m venv .venv
if errorlevel 1 (
    echo [LOI] Khong the tao moi truong ao .venv.
    exit /b 1
)
echo [2/2] Dang cai dat thu vien tu requirements.txt...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [LOI] Cai dat thu vien that bai.
    exit /b 1
)

:venv_ready
echo [Kiem tra] Chay test chong ro ri du lieu PaySim (test_no_leakage.py)...
"%PY%" code_paysim\tests\test_no_leakage.py
if errorlevel 1 (
    echo [LOI] Kiem tra ro ri du lieu that bai!
    exit /b 1
)

if "%~1"=="" goto :run_default

echo.
echo [Chay lenh] python run_all.py %*
echo ----------------------------------------------------------------------------
"%PY%" run_all.py %*
set "EXIT_CODE=%ERRORLEVEL%"
goto :finish

:run_default
echo.
echo [Chay toan bo] check + benchmark + robustness + ensemble + paysim + report (--resume)
echo ----------------------------------------------------------------------------
"%PY%" run_all.py all --resume
set "EXIT_CODE=%ERRORLEVEL%"

:finish
echo.
if "%EXIT_CODE%"=="0" (
    echo ============================================================================
    echo   HOAN TAT THANH CONG!
    echo   - Ket qua Benchmark  : ..\results\benchmark\
    echo   - Ket qua Robustness : ..\results\robustness\
    echo   - Ket qua Ensemble   : ..\results\ensemble\
    echo   - Ket qua PaySim     : code_paysim\results\
    echo ============================================================================
) else (
    echo ============================================================================
    echo   [CANH BAO] Tien trinh ket thuc voi ma loi: %EXIT_CODE%
    echo ============================================================================
)

exit /b %EXIT_CODE%
