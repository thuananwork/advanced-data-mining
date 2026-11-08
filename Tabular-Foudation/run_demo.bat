@echo off
setlocal EnableExtensions
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

cd /d "%~dp0"

echo ============================================================================
echo   KHOI DONG GIAO DIEN TUONG TAC (STREAMLIT DEMO)
echo   Mo trinh duyet tai: http://localhost:8501
echo ============================================================================

.\.venv\Scripts\python.exe -m streamlit run demo_app.py
