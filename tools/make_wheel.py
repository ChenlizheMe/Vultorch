"""Package a wheel from the compiled _vultorch extension module.

Called automatically by CMake (package_wheel target) after building _vultorch.
Can also be run manually:  python tools/make_wheel.py
"""

import hashlib, base64, zipfile, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VULTORCH_DIR = ROOT / "vultorch"
DIST_DIR = ROOT / "dist"


def get_version():
    toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"([^"]+)"', toml)
    if not m:
        raise RuntimeError("Cannot find version in pyproject.toml")
    return m.group(1)


def find_pyd():
    """Find the _vultorch extension module and return (path, wheel_tag)."""
    import sys
    # Prefer the platform-appropriate extension
    if sys.platform == "win32":
        preferred_suffix = ".pyd"
    else:
        preferred_suffix = ".so"

    candidates = []
    for f in VULTORCH_DIR.iterdir():
        if f.name.startswith("_vultorch") and f.suffix in (".pyd", ".so"):
            candidates.append(f)

    if not candidates:
        raise RuntimeError("No _vultorch.*.pyd/.so found in vultorch/")

    # Prefer the platform-matching extension
    candidates.sort(key=lambda f: (f.suffix != preferred_suffix, f.name))
    chosen = candidates[0]

    # Windows: _vultorch.cp310-win_amd64.pyd   → cp310-cp310-win_amd64
    # Linux:   _vultorch.cpython-310-x86_64-linux-gnu.so → cp310-cp310-linux_x86_64
    parts = chosen.stem.split(".", 1)
    if len(parts) == 2:
        raw = parts[1]
        if "linux" in raw:
            m = re.match(r"cpython-(\d+)-(\w+)-linux", raw)
            if m:
                pyver = f"cp{m.group(1)}"
                arch = m.group(2)
                tag = f"{pyver}-{pyver}-linux_{arch}"
            else:
                tag = raw
        else:
            segs = raw.split("-")
            if len(segs) == 2:
                tag = f"{segs[0]}-{segs[0]}-{segs[1]}"
            elif len(segs) >= 3:
                tag = raw
            else:
                tag = raw
    else:
        raise RuntimeError(f"Cannot parse platform tag from {chosen.name}")

    return chosen, tag


def _record_line(arc: str, data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    b64 = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return f"{arc},sha256={b64},{len(data)}"


def make_wheel():
    version = get_version()
    pyd_path, tag = find_pyd()
    whl_name = f"vultorch-{version}-{tag}.whl"
    DIST_DIR.mkdir(exist_ok=True)
    whl_path = DIST_DIR / whl_name

    if whl_path.exists():
        whl_path.unlink()

    dist_info = f"vultorch-{version}.dist-info"
    records = []

    with zipfile.ZipFile(whl_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── Package files ───────────────────────────────────────
        # Exclude extension modules for the wrong platform
        import sys
        skip_ext = ".so" if sys.platform == "win32" else ".pyd"
        for f in sorted(VULTORCH_DIR.iterdir()):
            if f.name.startswith("__pycache__"):
                continue
            if f.suffix == skip_ext and f.name.startswith("_vultorch"):
                continue
            if f.is_file():
                data = f.read_bytes()
                arc = f"vultorch/{f.name}"
                zf.writestr(arc, data)
                records.append(_record_line(arc, data))

        # ── METADATA ────────────────────────────────────────────
        readme = ROOT / "README.md"
        desc = readme.read_text(encoding="utf-8") if readme.exists() else ""
        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: vultorch\n"
            f"Version: {version}\n"
            "Summary: Real-time Torch visualization window with Vulkan zero-copy\n"
            "License: MIT\n"
            "Requires-Python: >=3.8\n"
            "Description-Content-Type: text/markdown\n"
            "\n" + desc
        )
        data = metadata.encode("utf-8")
        arc = f"{dist_info}/METADATA"
        zf.writestr(arc, data)
        records.append(_record_line(arc, data))

        # ── WHEEL ───────────────────────────────────────────────
        wheel_meta = (
            "Wheel-Version: 1.0\n"
            "Generator: vultorch-make-wheel\n"
            "Root-Is-Purelib: false\n"
            f"Tag: {tag}\n"
        )
        data = wheel_meta.encode()
        arc = f"{dist_info}/WHEEL"
        zf.writestr(arc, data)
        records.append(_record_line(arc, data))

        # ── top_level.txt ───────────────────────────────────────
        data = b"vultorch\n"
        arc = f"{dist_info}/top_level.txt"
        zf.writestr(arc, data)
        records.append(_record_line(arc, data))

        # ── RECORD ──────────────────────────────────────────────
        records.append(f"{dist_info}/RECORD,,")
        zf.writestr(f"{dist_info}/RECORD", "\n".join(records) + "\n")

    print(f"[vultorch] Wheel: dist/{whl_name}")
    return whl_path


if __name__ == "__main__":
    make_wheel()
