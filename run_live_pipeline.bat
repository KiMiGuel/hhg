@echo off
REM ============================================================
REM  HHG Face Identification & Blockchain Verification
REM  One-click LIVE pipeline runner for Windows.
REM
REM  Real SerpApi Google Lens + local Anvil blockchain + tamper drill.
REM
REM  Usage:
REM    Double-click this file in File Explorer, OR
REM    From cmd / PowerShell:  run_live_pipeline.bat
REM
REM  Optional environment variables (set before running):
REM    HHG_IMAGE   Override the input image (default: data\sample_face.jpg)
REM    HHG_REUSE=1 Skip the fresh-chain reset (reuse the existing Anvil + contract)
REM
REM  What it does (in order):
REM    1. cd into the project root so it works no matter how it is launched
REM    2. Verify venv, Anvil, and .env are present (with actionable errors)
REM    3. Show a pre-run banner and wait for the user to press a key
REM    4. Run:  python main.py live --image <img> [--fresh-chain]
REM    5. Print a clear success / failure banner
REM    6. Pause so the window stays open after a double-click launch
REM ============================================================

setlocal EnableExtensions EnableDelayedExpansion

REM --- 1. cd into the project root (directory this .bat lives in) ---
cd /d "%~dp0"

cls
echo.
echo ============================================================
echo   HHG Live Pipeline - one-click runner
echo.
echo   Project : %CD%
if defined HHG_IMAGE (
    echo   Image   : %HHG_IMAGE%
) else (
    echo   Image   : data\sample_face.jpg
)
if defined HHG_REUSE (
    echo   Mode    : REUSE existing chain and contract
) else (
    echo   Mode    : FRESH chain (auto-restart Anvil + redeploy)
)
echo ============================================================
echo.

REM --- 2. Verify the venv exists ---
if not exist "venv\Scripts\python.exe" goto :no_venv

REM --- 3. Verify Anvil is installed (Foundry) ---
set "ANVIL=%USERPROFILE%\.foundry\bin\anvil.exe"
if exist "%ANVIL%" goto :anvil_ok
where anvil.exe >nul 2>&1
if not errorlevel 1 goto :anvil_ok
goto :no_anvil

:no_venv
color 0C
echo [ERROR] venv\Scripts\python.exe not found in %CD%.
echo.
echo         Bootstrap with:
echo             python -m venv venv
echo             venv\Scripts\pip install -r requirements.txt
echo.
pause
endlocal & exit /b 1

:no_anvil
color 0C
echo [ERROR] anvil.exe not found at %ANVIL% and not on PATH.
echo.
echo         Install Foundry from:
echo             https://book.getfoundry.sh/getting-started/installation
echo.
pause
endlocal & exit /b 1

:anvil_ok

REM --- 4. Verify .env exists ---
if exist ".env" goto :env_ok
color 0C
echo [ERROR] .env not found in %CD%.
echo.
echo         Copy .env.example to .env and fill in SERPAPI_KEY.
echo         Get a free key (no credit card) at https://serpapi.com
echo.
pause
endlocal & exit /b 1

:env_ok

REM --- 5. Build the flags ---
set "IMG=data\sample_face.jpg"
if defined HHG_IMAGE set "IMG=%HHG_IMAGE%"

set "FLAGS=live --image "%IMG%""
if not defined HHG_REUSE set "FLAGS=%FLAGS% --fresh-chain"

echo Pipeline will:
echo   1. (Re)start Anvil at http://127.0.0.1:8545
echo   2. Deploy FaceRegistry.sol (if needed) and update .env
echo   3. Detect face, generate SHA-256 biometric hash
echo   4. Upload crop to catbox.moe and run SerpApi Google Lens
echo   5. Anchor the result on-chain
echo   6. Re-verify, then run the tamper-evidence drill
echo.
echo Image file: %IMG%
if exist "%IMG%" goto :img_ok
color 0C
echo.
echo [ERROR] Image not found: %IMG%
echo         Set HHG_IMAGE=path\to\your_face.jpg and re-run.
echo.
pause
endlocal & exit /b 1

:img_ok
echo.
echo Press any key to start the LIVE pipeline (Ctrl+C to cancel)...
pause >nul
echo.

REM --- 6. Run the pipeline; capture the exit code ---
color 0A
venv\Scripts\python.exe main.py %FLAGS%
set "EXITCODE=%ERRORLEVEL%"

echo.
if not "%EXITCODE%"=="0" goto :run_failed
color 0A
echo ============================================================
echo   SUCCESS - pipeline completed end-to-end.
echo.
echo   Audit report (JSON): reports\report_*.json
echo   Audit report (MD)  : reports\report_*.md
echo ============================================================
goto :run_done

:run_failed
color 0C
echo ============================================================
echo   FAILED (exit code %EXITCODE%).
echo.
echo   Common causes:
echo     - SERPAPI_KEY missing or out of credits (check .env)
echo     - Internet blocked (serpapi.com / catbox.moe / tmpfiles.org)
echo     - Anvil not installed (need Foundry)
echo.
echo   Re-run:  python main.py setup
echo ============================================================

:run_done

REM --- 7. Pause so the window doesn't auto-close on double-click ---
echo.
pause
endlocal & exit /b %EXITCODE%
