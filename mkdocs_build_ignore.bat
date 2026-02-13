@echo off
setlocal

REM One-click MkDocs build (ignore failures and keep CI/local scripts alive)
cd /d %~dp0

where mkdocs >nul 2>nul
if errorlevel 1 (
  echo [mkdocs] not found, skipping tutorial build.
  exit /b 0
)

echo [mkdocs] building tutorial HTML...
mkdocs build --clean
if errorlevel 1 (
  echo [mkdocs] build failed, ignored by mkdocs_build_ignore.bat
  exit /b 0
)

echo [mkdocs] done. Output: docs\tutorial\index.html
exit /b 0
