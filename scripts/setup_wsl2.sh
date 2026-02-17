#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  scripts/setup_wsl2.sh — Set up WSL2 Ubuntu for building Vultorch
#
#  Run inside WSL2:
#    cd /mnt/d/Vultorch
#    sudo bash scripts/setup_wsl2.sh
# ═══════════════════════════════════════════════════════════════
set -e

echo "════════════════════════════════════════"
echo "  Vultorch WSL2 Build Environment Setup"
echo "════════════════════════════════════════"

# ── System packages ────────────────────────────────────────────
echo "[1/4] Installing system dependencies ..."
apt-get update -qq
apt-get install -y -qq \
    build-essential cmake ninja-build \
    libvulkan-dev vulkan-tools \
    python3-dev python3-pip python3-venv \
    libsdl2-dev libx11-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev libwayland-dev \
    glslang-tools

# ── Python packages ────────────────────────────────────────────
echo "[2/4] Installing Python build tools ..."
pip3 install --break-system-packages scikit-build-core pybind11 build 2>/dev/null \
    || pip3 install scikit-build-core pybind11 build

# ── Verify Vulkan ──────────────────────────────────────────────
echo "[3/4] Verifying Vulkan ..."
if pkg-config --exists vulkan 2>/dev/null; then
    echo "  Vulkan: $(pkg-config --modversion vulkan)"
else
    echo "  Vulkan headers found at /usr/include/vulkan/"
fi

# ── CUDA check (optional) ─────────────────────────────────────
echo "[4/4] Checking CUDA ..."
if command -v nvcc &>/dev/null; then
    echo "  CUDA: $(nvcc --version | grep release)"
elif [ -d /usr/local/cuda ]; then
    echo "  CUDA found at /usr/local/cuda (add to PATH)"
else
    echo "  CUDA not found — building CPU-only (OK)"
fi

echo ""
echo "════════════════════════════════════════"
echo "  Setup complete! Build with:"
echo "    cd /mnt/d/Vultorch"
echo "    cmake --preset release-linux"
echo "    cmake --build --preset release-linux"
echo "════════════════════════════════════════"
