@echo off
REM MuniRadar startup (Windows) - double-click to launch
REM Keys live in muniradar-secrets.bat next to this file (never committed to git)
setlocal
cd /d "%~dp0"

if not exist "muniradar-secrets.bat" (
  echo [X] muniradar-secrets.bat not found in %~dp0
  echo     Copy it from the Mac folder or create it with your keys.
  pause
  exit /b 1
)
call "muniradar-secrets.bat"

REM Pick Python launcher
set "PY=python"
where py >nul 2>&1 && set "PY=py -3"

REM Pick ngrok (PATH or ngrok.exe in this folder)
set "NGROK=ngrok"
if exist "%~dp0ngrok.exe" set "NGROK=%~dp0ngrok.exe"

REM First run: install Python deps if missing
%PY% -c "import fastapi, uvicorn, httpx" >nul 2>&1 || %PY% -m pip install -r requirements.txt
%PY% -c "import blpapi" >nul 2>&1 || echo [!] blpapi not importable - Bloomberg lookups will fail

REM Kill anything already on port 8001
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

REM Start backend in its own window
start "MuniRadar Backend" cmd /k %PY% -m uvicorn main:app --host 127.0.0.1 --port 8001
echo [OK] Backend starting on port 8001
echo Opening https://muniradar.netlify.app in 3 seconds...
timeout /t 3 /nobreak >nul
start "" "https://muniradar.netlify.app"

REM Start ngrok (foreground - keeps this window open)
"%NGROK%" config add-authtoken %NGROK_AUTHTOKEN% >nul
"%NGROK%" http --domain=muniradar.ngrok.app 8001
