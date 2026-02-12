<#
.SYNOPSIS
    Build vultorch for the current Python environment.

.DESCRIPTION
    Default: builds a wheel and installs it via pip.
    -Dev:    fast incremental cmake build (for development iteration).
    -NoInstall: build wheel only, skip pip install.

.EXAMPLE
    .\build.ps1                # build wheel + install
    .\build.ps1 -Dev           # fast cmake dev build
    .\build.ps1 -NoInstall     # build wheel only
#>

param(
    [switch]$Dev,
    [switch]$NoInstall
)

$ErrorActionPreference = "Stop"

$pyVer = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pyTag = "cp$($pyVer -replace '\.','')"
Write-Host "[vultorch] Python $pyVer ($pyTag)" -ForegroundColor Cyan

if ($Dev) {
    # ── Fast dev build: cmake only ──────────────────────────────
    if (-not (Test-Path "build/release-windows/build.ninja")) {
        Write-Host "[vultorch] Configuring CMake ..." -ForegroundColor Cyan
        cmake --preset release-windows
        if ($LASTEXITCODE -ne 0) { Write-Error "CMake configure failed"; exit 1 }
    }
    Write-Host "[vultorch] Building ..." -ForegroundColor Cyan
    cmake --build build/release-windows --config Release
    if ($LASTEXITCODE -ne 0) { Write-Error "Build failed"; exit 1 }

    # Copy .pyd for editable install
    $pyd = Get-ChildItem "build/release-windows/_vultorch.*$pyTag*.pyd" -ErrorAction SilentlyContinue |
           Select-Object -First 1
    if ($pyd) {
        Copy-Item $pyd.FullName "vultorch/" -Force
        Write-Host "[vultorch] $($pyd.Name) -> vultorch/" -ForegroundColor Green
    }
    Write-Host "[vultorch] Dev build complete." -ForegroundColor Green
    Write-Host "  Run 'pip install -e .' once for editable install." -ForegroundColor DarkGray
}
else {
    # ── Wheel build ─────────────────────────────────────────────
    $distDir = Join-Path $PSScriptRoot "dist"
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null

    Write-Host "[vultorch] Building wheel ..." -ForegroundColor Cyan
    python -m build --wheel --outdir $distDir
    if ($LASTEXITCODE -ne 0) { Write-Error "Wheel build failed"; exit 1 }

    $whl = Get-ChildItem "$distDir\vultorch-*$pyTag*.whl" |
           Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if (-not $whl) {
        Write-Error "No wheel found matching $pyTag"
        exit 1
    }

    Write-Host "[vultorch] Wheel: $($whl.Name)" -ForegroundColor Green

    if (-not $NoInstall) {
        Write-Host "[vultorch] Installing ..." -ForegroundColor Cyan
        pip install $whl.FullName --force-reinstall -q
        python -c "import vultorch; print(f'[vultorch] Installed v{vultorch.__version__} OK')"
    }
}
