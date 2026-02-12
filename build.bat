@echo off
setlocal
REM ═══════════════════════════════════════════════════════════
REM  build.bat — Configure + build + produce wheel in dist/
REM  Usage:  build.bat           (full build + wheel)
REM          build.bat --dev     (cmake build only, no wheel repackage)
REM ═══════════════════════════════════════════════════════════

if "%1"=="--dev" goto :dev

:full
if not exist "build\release-windows\build.ninja" (
    echo [vultorch] Configuring ...
    cmake --preset release-windows
    if errorlevel 1 exit /b 1
)
echo [vultorch] Building + packaging wheel ...
cmake --build --preset release-windows
if errorlevel 1 exit /b 1
echo [vultorch] Done.  Wheel in dist\
goto :eof

:dev
if not exist "build\release-windows\build.ninja" (
    echo [vultorch] Configuring ...
    cmake --preset release-windows
    if errorlevel 1 exit /b 1
)
echo [vultorch] Building (dev) ...
cmake --build --preset release-windows
if errorlevel 1 exit /b 1
echo [vultorch] Done.
goto :eof
