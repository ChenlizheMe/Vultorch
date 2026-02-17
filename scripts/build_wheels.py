#!/usr/bin/env python3
"""Build wheels for multiple Python versions using conda environments.

Usage:
    python scripts/build_wheels.py                   # default versions
    python scripts/build_wheels.py 3.10 3.11 3.12    # specific versions

Requires conda to be available on PATH.
"""

import subprocess
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"

DEFAULT_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kwargs)


def main():
    versions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_VERSIONS
    DIST.mkdir(exist_ok=True)
    preset = "release-windows" if platform.system() == "Windows" else "release-linux"

    for ver in versions:
        env_name = f"vultorch-build-{ver}"
        print(f"\n{'═' * 48}")
        print(f"  Building wheel for Python {ver}")
        print(f"{'═' * 48}")

        # Create / reuse conda env
        subprocess.run(
            ["conda", "create", "-n", env_name, f"python={ver}", "-y", "-q"],
            capture_output=True,
        )

        # Get the Python executable path
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import sys; print(sys.executable)"],
            capture_output=True, text=True,
        )
        py_exe = result.stdout.strip()
        if not py_exe:
            print(f"  SKIP: could not resolve Python {ver}")
            continue
        print(f"  Python: {py_exe}")

        # Remove stale extension modules
        for ext in ("*.pyd", "*.so"):
            for f in (ROOT / "vultorch").glob(f"_vultorch{ext}"):
                f.unlink(missing_ok=True)

        # Configure + build
        try:
            run(["cmake", "--preset", preset, "--fresh",
                 f"-DPython3_EXECUTABLE={py_exe}"])
            run(["cmake", "--build", "--preset", preset, "--clean-first"])
            print(f"  OK: Python {ver}")
        except subprocess.CalledProcessError:
            print(f"  FAILED: Python {ver}")

    print(f"\n{'═' * 48}")
    print("  Wheels in dist/")
    for whl in sorted(DIST.glob("*.whl")):
        print(f"    {whl.name}")
    print(f"{'═' * 48}")


if __name__ == "__main__":
    main()
