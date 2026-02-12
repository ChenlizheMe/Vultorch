<#
.SYNOPSIS
    Uploads vultorch wheels in dist/ to PyPI using Twine.

.DESCRIPTION
    This script checks for the 'twine' tool, installs it if missing,
    and then uploads all .whl files found in the dist/ directory to PyPI.
    It prompts for the PyPI API Token securely.

.EXAMPLE
    .\upload_wheels.ps1
#>

$ErrorActionPreference = "Stop"
$distDir = Join-Path $PSScriptRoot "dist"

# Check if dist dir exists and has wheels
if (-not (Test-Path $distDir) -or (Get-ChildItem "$distDir\*.whl").Count -eq 0) {
    Write-Error "No wheels found in '$distDir'. Please run build_wheels.ps1 first."
}

# Check for twine
if (-not (Get-Command "twine" -ErrorAction SilentlyContinue)) {
    Write-Host "Twine not found. Installing..." -ForegroundColor Yellow
    pip install twine
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install twine."
    }
}

Write-Host "`nFound the following wheels to upload:" -ForegroundColor Cyan
Get-ChildItem "$distDir\*.whl" | ForEach-Object { Write-Host "  - $($_.Name)" }

Write-Host "`nTo upload to PyPI, you need an API Token."
Write-Host "1. Go to https://pypi.org/manage/account/token/"
Write-Host "2. Create a token (Scope: Entire account)"
Write-Host "3. Copy the token (starts with pypi-...)" -ForegroundColor Yellow

$token = Read-Host -Prompt "Paste your PyPI API Token (pypi-...)" -AsSecureString

if (-not $token) {
    Write-Error "Token is required to proceed."
}

# Convert SecureString back to plain text for the command (only lives in memory for command execution)
$tokenPlain = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))

Write-Host "`nUploading to PyPI..." -ForegroundColor Green

# Invoke twine
# --repository pypi is default, but explicit is good
# -u __token__ is the username for token auth
# -p $tokenPlain is the password
twine upload --repository pypi -u __token__ -p $tokenPlain "$distDir\*.whl"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS! Your package is now on PyPI." -ForegroundColor Green
    Write-Host "You can install it with: pip install vultorch" -ForegroundColor Cyan
} else {
    Write-Host "`nUpload failed. Please check the error message above." -ForegroundColor Red
    Write-Host "Common reasons: Package name already taken, version already exists, or invalid token."
}
