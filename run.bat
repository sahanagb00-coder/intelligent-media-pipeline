@echo off
cd /d "%~dp0"
python -m uvicorn main:app --reload
pause