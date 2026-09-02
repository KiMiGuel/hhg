# Starts a local Anvil EVM node for zero-cost, zero-gas demos.
# Foundry binaries are expected at %USERPROFILE%\.foundry\bin
$anvil = Join-Path $env:USERPROFILE ".foundry\bin\anvil.exe"
if (-not (Test-Path $anvil)) {
    Write-Error "anvil.exe not found at $anvil. Install Foundry first."
    exit 1
}
Write-Host "Starting Anvil at http://127.0.0.1:8545 (Ctrl+C to stop)..." -ForegroundColor Cyan
& $anvil --host 127.0.0.1 --port 8545
