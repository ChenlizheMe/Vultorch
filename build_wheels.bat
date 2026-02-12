@echo off
setlocal enabledelayedexpansion
REM ═══════════════════════════════════════════════════════════
REM  build_wheels.bat — Build wheels for multiple Python versions
REM  Requires conda.  Usage:  build_wheels.bat [3.9 3.10 3.11 3.12]
REM ═══════════════════════════════════════════════════════════

if "%~1"=="" (
    set VERSIONS=3.8 3.9 3.10 3.11 3.12
) else (
    set VERSIONS=%*
)

if not exist dist mkdir dist

for %%V in (%VERSIONS%) do (
    echo.
    echo ════════════════════════════════════════
    echo   Building wheel for Python %%V
    echo ════════════════════════════════════════

    call conda create -n vultorch-build-%%V python=%%V -y -q 2>nul

    REM Get the Python executable path from the conda env
    for /f "delims=" %%P in ('conda run -n vultorch-build-%%V python -c "import sys; print(sys.executable)"') do set "PY_EXE=%%P"
    echo   Python: !PY_EXE!

    REM Remove stale .pyd so make_wheel picks the freshly built one
    del /q "%~dp0vultorch\_vultorch*.pyd" 2>nul
    del /q "%~dp0vultorch\_vultorch*.so"  2>nul

    REM Fresh configure + clean build with explicit Python path
    cmake --preset release-windows --fresh -DPython3_EXECUTABLE="!PY_EXE!"
    if errorlevel 1 (
        echo   CONFIGURE FAILED for Python %%V
    ) else (
        cmake --build --preset release-windows --clean-first
        if errorlevel 1 (
            echo   BUILD FAILED for Python %%V
        ) else (
            echo   OK for Python %%V
        )
    )
)

echo.
echo ════════════════════════════════════════
echo   Wheels in dist\
dir /b dist\*.whl
echo ════════════════════════════════════════
