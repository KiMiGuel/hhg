# Resets the demo to a clean state for recording:
#  1. Stops any running Anvil
#  2. Launches a fresh Anvil node (visible window - keep it on screen!)
#  3. Redeploys FaceRegistry and updates CONTRACT_ADDRESS in .env
# Then run:  python pipeline.py data/sample_face.jpg

Write-Host "Stopping any existing Anvil..." -ForegroundColor Cyan
Get-Process anvil -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

$anvil = Join-Path $env:USERPROFILE ".foundry\bin\anvil.exe"
if (-not (Test-Path $anvil)) { Write-Error "anvil.exe not found at $anvil"; exit 1 }

Write-Host "Launching fresh Anvil at http://127.0.0.1:8545 ..." -ForegroundColor Cyan
Start-Process -FilePath $anvil -ArgumentList "--host", "127.0.0.1", "--port", "8545"
Start-Sleep -Seconds 3

Write-Host "Deploying FaceRegistry..." -ForegroundColor Cyan
$addr = & ".\venv\Scripts\python.exe" scripts\deploy.py | Select-String -Pattern "Contract Address: (\S+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
if (-not $addr) { Write-Error "Deployment failed."; exit 1 }

Write-Host "Updating .env with CONTRACT_ADDRESS=$addr" -ForegroundColor Green
$envPath = Join-Path $PSScriptRoot "..\.env"
$lines = Get-Content $envPath
$found = $false
$lines = $lines | ForEach-Object {
    if ($_ -match '^CONTRACT_ADDRESS=') { $found = $true; "CONTRACT_ADDRESS=`"$addr`"" } else { $_ }
}
if (-not $found) { $lines += "CONTRACT_ADDRESS=`"$addr`"" }
Set-Content $envPath $lines

Write-Host "`nREADY! Record your demo now:  python pipeline.py data/sample_face.jpg" -ForegroundColor Green
