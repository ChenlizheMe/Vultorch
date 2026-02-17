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


def _env_candidates(ver: str) -> list[str]:
    compact = ver.replace(".", "")
    return [f"vultorch-build-{compact}", f"vultorch-build-{ver}"]


def _resolve_python_from_env(env_name: str) -> str:
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-c",
         "import sys; print(sys.executable)"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return ""

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1]


def main():
    versions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_VERSIONS
    DIST.mkdir(exist_ok=True)
    preset = "release-windows" if platform.system() == "Windows" else "release-linux"

    for ver in versions:
        env_candidates = _env_candidates(ver)
        preferred_env = env_candidates[0]
        print(f"\n{'═' * 48}")
        print(f"  Building wheel for Python {ver}")
        print(f"{'═' * 48}")

        # First try existing known env-name patterns (e.g. -38 and -3.8)
        py_exe = ""
        matched_env = ""
        for env_name in env_candidates:
            py_exe = _resolve_python_from_env(env_name)
            if py_exe:
                matched_env = env_name
                break

        # If none works, create preferred env name then retry once
        if not py_exe:
            create = subprocess.run(
                ["conda", "create", "-n", preferred_env, f"python={ver}", "-y", "-q"],
                capture_output=True, text=True,
            )
            if create.returncode != 0:
                err = (create.stderr or create.stdout or "").strip()
                if err:
                    print("  conda create error:")
                    print(f"    {err.splitlines()[-1]}")

            py_exe = _resolve_python_from_env(preferred_env)
            matched_env = preferred_env if py_exe else ""

        if not py_exe:
            print(f"  SKIP: could not resolve Python {ver}")
            continue
        print(f"  Env: {matched_env}")
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
