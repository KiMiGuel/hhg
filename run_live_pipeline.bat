@echo off
REM ============================================================
REM   HHG  ::  FACE IDENTITY + BLOCKCHAIN VERIFICATION
REM   One-click LIVE pipeline (real SerpApi + local Anvil).
REM
REM   Interactive:    double-click, or run from cmd.
REM   Non-interactive: set HHG_NONINTERACTIVE=1 (defaults used).
REM   Env vars:       HHG_IMAGE=path  HHG_REUSE=1  HHG_NONINTERACTIVE=1
REM ============================================================
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

color 0B
cls
echo.
echo   ===================================================
echo     HHG  ::  FACE IDENTITY + BLOCKCHAIN PROOF
echo   ===================================================
echo.

if defined HHG_NONINTERACTIVE goto :apply_defaults

REM ---- Interactive: print the input menu ----
echo   [1]  Use bundled sample (data\sample_face.jpg)
echo   [2]  Capture from webcam
echo   [3]  Pick another image file
echo   [4]  Cancel
echo.
set "INPUT_MODE="
set /p "INPUT_MODE=Choose input [1-4, default 1]: "
if not defined INPUT_MODE set "INPUT_MODE=1"
if "%INPUT_MODE%"=="" set "INPUT_MODE=1"

if "%INPUT_MODE%"=="4" goto :user_cancel

if "%INPUT_MODE%"=="1" goto :pick_sample
if "%INPUT_MODE%"=="2" goto :pick_webcam
if "%INPUT_MODE%"=="3" goto :pick_other
echo Invalid choice: %INPUT_MODE%
pause
endlocal & exit /b 1

:pick_sample
if not exist "data\sample_face.jpg" goto :no_sample
set "INPUT_TARGET=data\sample_face.jpg"
set "INPUT_FLAG=--image"
goto :ask_chain_mode

:pick_webcam
REM For --camera the path is the OUTPUT snapshot location; we pass it
REM to Python as --output, NOT as a positional.
set "INPUT_TARGET=data\captured_face.jpg"
set "INPUT_FLAG=--camera"
goto :ask_chain_mode

:pick_other
set "INPUT_TARGET="
set /p "INPUT_TARGET=Path to image: "
if not exist "%INPUT_TARGET%" (
    color 0C
    echo.
    echo [ERROR] Image not found: %INPUT_TARGET%
    pause
    endlocal & exit /b 1
)
set "INPUT_FLAG=--image"
goto :ask_chain_mode

:no_sample
color 0C
echo.
echo [ERROR] data\sample_face.jpg not found.
pause
endlocal & exit /b 1

:user_cancel
echo Cancelled.
pause
endlocal & exit /b 0

:ask_chain_mode
echo.
echo   [F]  Fresh chain (auto-restart Anvil + redeploy)
echo   [R]  Reuse existing chain and contract
echo.
set "CHAIN_MODE="
set /p "CHAIN_MODE=Chain mode [F/R, default F]: "
if not defined CHAIN_MODE set "CHAIN_MODE=F"
if /I "%CHAIN_MODE%"=="F" set "CHAIN_FRESH=1"
if /I "%CHAIN_MODE%"=="R" set "CHAIN_FRESH=0"
if /I not "%CHAIN_MODE%"=="F" if /I not "%CHAIN_MODE%"=="R" set "CHAIN_FRESH=1"
goto :preflight

:apply_defaults
set "INPUT_MODE=1"
set "INPUT_TARGET=data\sample_face.jpg"
set "INPUT_FLAG=--image"
set "CHAIN_FRESH=1"

:preflight
REM ---- Apply optional env-var overrides ----
if defined HHG_IMAGE set "INPUT_TARGET=%HHG_IMAGE%"
if /I "%HHG_REUSE%"=="1" set "CHAIN_FRESH=0"
if not defined CHAIN_FRESH set "CHAIN_FRESH=1"

REM ---- Build the chain flag ----
if "%CHAIN_FRESH%"=="1" (
    set "CHAIN_ARGS=--fresh-chain"
    set "CHAIN_LABEL=fresh chain (auto-restart Anvil + redeploy)"
) else (
    set "CHAIN_ARGS="
    set "CHAIN_LABEL=reuse existing chain and contract"
)

REM ---- Preflight (fast) ----
if not exist "venv\Scripts\python.exe" goto :no_venv
set "ANVIL=%USERPROFILE%\.foundry\bin\anvil.exe"
if exist "%ANVIL%" goto :anvil_ok
where anvil.exe >nul 2>&1
if not errorlevel 1 goto :anvil_ok
goto :no_anvil

:no_venv
color 0C
echo.
echo [ERROR] venv missing. Run: python -m venv venv ^&^& venv\Scripts\pip install -r requirements.txt
pause
endlocal & exit /b 1

:no_anvil
color 0C
echo.
echo [ERROR] anvil.exe not found. Install Foundry: https://book.getfoundry.sh/getting-started/installation
pause
endlocal & exit /b 1

:anvil_ok
if exist ".env" goto :env_ok
color 0C
echo.
echo [ERROR] .env missing. Copy .env.example to .env and add SERPAPI_KEY.
pause
endlocal & exit /b 1

:env_ok
echo.
echo   ===================================================
echo     STARTING LIVE PIPELINE
echo     input   : %INPUT_TARGET%
echo     chain   : %CHAIN_LABEL%
echo   ===================================================
echo.

REM ---- Run, capture log to a unique path ----
set "LOG=%TEMP%\hhg_live_%RANDOM%.log"
color 0A
if "%INPUT_FLAG%"=="--camera" (
    venv\Scripts\python.exe main.py live --camera --output "%INPUT_TARGET%" %CHAIN_ARGS% > "%LOG%" 2>&1
) else (
    venv\Scripts\python.exe main.py live --image "%INPUT_TARGET%" %CHAIN_ARGS% > "%LOG%" 2>&1
)
set "EXITCODE=%ERRORLEVEL%"

echo.
if "%EXITCODE%"=="0" goto :run_ok
color 0C
echo   ===================================================
echo     FAILED  (exit code %EXITCODE%)
echo   ===================================================
echo.
echo   ---- last 30 lines of the run ----
powershell -NoProfile -Command "Get-Content -LiteralPath '%LOG%' -Tail 30" 2>nul
echo   ----------------------------------
echo.
echo   Full log: %LOG%
goto :finish

:run_ok
color 0A
echo   ===================================================
echo     SUCCESS  -  pipeline completed end-to-end
echo     audit   : reports\report_*.json
echo     report : reports\report_*.md
echo   ===================================================

:finish
echo.
if defined HHG_NONINTERACTIVE (
    endlocal & exit /b %EXITCODE%
) else (
    pause
    endlocal & exit /b %EXITCODE%
)
