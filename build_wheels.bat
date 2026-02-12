@echo off
setlocal
REM ═══════════════════════════════════════════════════════════
REM  build_wheels.bat — Build wheels for multiple Python versions
REM  Requires conda.  Usage:  build_wheels.bat [3.9 3.10 3.11 3.12]
REM ═══════════════════════════════════════════════════════════

if "%~1"=="" (
    set VERSIONS=3.9 3.10 3.11 3.12
) else (
    set VERSIONS=%*
)

if not exist dist mkdir dist

for %%V in (%VERSIONS%) do (
    set "VER=%%V"
    set "ENV=vultorch-build-%%V"
    echo.
    echo ════════════════════════════════════════
    echo   Building wheel for Python %%V
    echo ════════════════════════════════════════

    call conda create -n vultorch-build-%%V python=%%V -y -q 2>nul
    call conda run -n vultorch-build-%%V pip install -q scikit-build-core pybind11 build

    call conda run -n vultorch-build-%%V python -m build --wheel --outdir dist
    if errorlevel 1 (
        echo   FAILED for Python %%V
    ) else (
        echo   OK for Python %%V
    )
)

echo.
echo ════════════════════════════════════════
echo   Wheels in dist\
dir /b dist\*.whl
echo ════════════════════════════════════════
