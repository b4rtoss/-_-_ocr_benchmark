@echo off
cd /d "%~dp0"
py -3.12 -m venv .venv
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r manga_app\requirements.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto failed
echo Installation complete. Open the launcher.
pause
exit /b 0
:failed
echo Installation failed. Python 3.12 and internet access are required.
pause
exit /b 1
