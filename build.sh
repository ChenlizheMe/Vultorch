#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  build.sh — Configure + build + produce wheel in dist/
#  Usage:  ./build.sh           (full build + wheel)
#          ./build.sh --dev     (cmake build only)
# ═══════════════════════════════════════════════════════════
set -e

PRESET="release-linux"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PRESET="release-windows"
fi

if [ ! -f "build/${PRESET}/build.ninja" ]; then
    echo "[vultorch] Configuring ..."
    cmake --preset "$PRESET"
fi

echo "[vultorch] Building ..."
cmake --build --preset "$PRESET"

# Build tutorial docs (mkdocs)
if command -v mkdocs &>/dev/null; then
    echo "[vultorch] Building tutorial docs ..."
    mkdocs build
else
    echo "[vultorch] mkdocs not found, skipping tutorial build."
    echo "           Install with: pip install mkdocs-material mkdocs-static-i18n"
fi

echo "[vultorch] Done.  Wheel in dist/"
