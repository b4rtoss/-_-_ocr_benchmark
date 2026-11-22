@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please run Install.cmd first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m manga_app.app
if errorlevel 1 pause
