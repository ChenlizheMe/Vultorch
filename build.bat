@echo off
setlocal
REM ═══════════════════════════════════════════════════════════
REM  build.bat — Configure + build + produce wheel in dist/
REM  Usage:  build.bat           (full clean build + wheel)
REM          build.bat --dev     (incremental build, no reconfigure)
REM
REM  Default mode always reconfigures from scratch so the wheel
REM  matches whichever Python is currently active.
REM ═══════════════════════════════════════════════════════════

REM Resolve the active Python executable
for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PY_EXE=%%P"
    goto :found_python
)
echo [vultorch] ERROR: python not found on PATH
exit /b 1
:found_python
echo [vultorch] Using Python: %PY_EXE%

if "%1"=="--dev" goto :dev

:full
echo [vultorch] Configuring (fresh) ...
REM Remove stale .pyd files so make_wheel picks the freshly built one
del /q "%~dp0vultorch\_vultorch*.pyd" 2>nul
del /q "%~dp0vultorch\_vultorch*.so"  2>nul
cmake --preset release-windows --fresh -DPython3_EXECUTABLE="%PY_EXE%"
if errorlevel 1 exit /b 1
echo [vultorch] Building + packaging wheel ...
cmake --build --preset release-windows --clean-first
if errorlevel 1 exit /b 1
echo [vultorch] Done.  Wheel in dist\
goto :eof

:dev
if not exist "build\release-windows\build.ninja" (
    echo [vultorch] Configuring ...
    cmake --preset release-windows -DPython3_EXECUTABLE="%PY_EXE%"
    if errorlevel 1 exit /b 1
)
echo [vultorch] Building (dev, incremental) ...
cmake --build --preset release-windows
if errorlevel 1 exit /b 1
echo [vultorch] Done.
goto :eof
