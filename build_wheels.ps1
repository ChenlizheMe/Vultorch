<#
.SYNOPSIS
    Build vultorch wheels for multiple Python versions using conda.

.EXAMPLE
    .\build_wheels.ps1
    .\build_wheels.ps1 -Versions "3.10","3.12"
#>

param(
    [string[]]$Versions = @("3.9", "3.10", "3.11", "3.12")
)

$ErrorActionPreference = "Stop"
$distDir = Join-Path $PSScriptRoot "dist"
New-Item -ItemType Directory -Force -Path $distDir | Out-Null

foreach ($ver in $Versions) {
    $envName = "vultorch-build-$($ver -replace '\.', '')"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Building wheel for Python $ver ($envName)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Create or update the build environment
    conda create -n $envName python=$ver -y -q 2>$null
    # Update SSL/certificates for older Pythons, then install build deps
    conda install -n $envName -y -q certifi ca-certificates openssl pip 2>$null
    conda run -n $envName pip install -q --trusted-host pypi.org --trusted-host files.pythonhosted.org scikit-build-core pybind11 build

    # Build wheel
    Push-Location $PSScriptRoot
    try {
        conda run -n $envName python -m build --wheel --outdir $distDir
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  FAILED for Python $ver" -ForegroundColor Red
            continue
        }
    } finally {
        Pop-Location
    }

    # Quick smoke test
    $whl = Get-ChildItem "$distDir\vultorch-*cp$($ver -replace '\.', '')*win_amd64.whl" |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($whl) {
        conda run -n $envName pip install $whl.FullName --force-reinstall -q
        conda run -n $envName python -c "import vultorch; print('  OK:', vultorch.__version__)"
        Write-Host "  Wheel: $($whl.Name)" -ForegroundColor Green
    }
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  All wheels:" -ForegroundColor Cyan
Get-ChildItem "$distDir\*.whl" | ForEach-Object { Write-Host "    $_" }
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nTo upload to PyPI:" -ForegroundColor Yellow
Write-Host "  pip install twine" -ForegroundColor Yellow
Write-Host "  twine upload dist/*.whl" -ForegroundColor Yellow
