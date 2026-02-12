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

echo "[vultorch] Done.  Wheel in dist/"
