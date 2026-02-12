#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  build_wheels.sh — Build wheels for multiple Python versions
#  Requires conda.  Usage:  ./build_wheels.sh [3.9 3.10 3.11 3.12]
# ═══════════════════════════════════════════════════════════
set -e

VERSIONS="${@:-3.9 3.10 3.11 3.12}"
mkdir -p dist

for VER in $VERSIONS; do
    ENV="vultorch-build-${VER}"
    echo ""
    echo "════════════════════════════════════════"
    echo "  Building wheel for Python ${VER}"
    echo "════════════════════════════════════════"

    conda create -n "$ENV" python="$VER" -y -q 2>/dev/null || true
    conda run -n "$ENV" pip install -q scikit-build-core pybind11 build
    conda run -n "$ENV" python -m build --wheel --outdir dist || {
        echo "  FAILED for Python ${VER}"
        continue
    }
    echo "  OK for Python ${VER}"
done

echo ""
echo "════════════════════════════════════════"
echo "  Wheels in dist/"
ls -1 dist/*.whl 2>/dev/null
echo "════════════════════════════════════════"
