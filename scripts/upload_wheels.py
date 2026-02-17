#!/usr/bin/env python3
"""Upload vultorch wheels from dist/ to PyPI using twine.

Usage:
    python scripts/upload_wheels.py                # interactive token prompt
    python scripts/upload_wheels.py --token TOKEN  # pass token directly

Requires twine: pip install twine
"""

import subprocess
import sys
import getpass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"


def main():
    wheels = sorted(DIST.glob("*.whl"))
    if not wheels:
        print(f"ERROR: No wheels found in {DIST}. Build first with:")
        print("  cmake --preset <preset>")
        print("  cmake --build --preset <preset>")
        sys.exit(1)

    # Ensure twine is installed
    try:
        subprocess.run(["twine", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("twine not found, installing ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "twine"])

    print(f"\nFound {len(wheels)} wheel(s) to upload:")
    for w in wheels:
        print(f"  {w.name}")

    # Get token
    token = None
    if "--token" in sys.argv:
        idx = sys.argv.index("--token")
        if idx + 1 < len(sys.argv):
            token = sys.argv[idx + 1]

    if not token:
        print("\nTo upload to PyPI, you need an API Token.")
        print("  1. Go to https://pypi.org/manage/account/token/")
        print("  2. Create a token (Scope: project 'vultorch')")
        print("  3. Paste the token below (starts with pypi-...)")
        token = getpass.getpass("PyPI API Token: ")

    if not token:
        print("ERROR: Token is required.")
        sys.exit(1)

    print("\nUploading to PyPI ...")
    result = subprocess.run([
        "twine", "upload",
        "--repository", "pypi",
        "-u", "__token__",
        "-p", token,
    ] + [str(w) for w in wheels])

    if result.returncode == 0:
        print("\nSUCCESS — install with: pip install vultorch")
    else:
        print("\nUpload failed. Check the error above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
