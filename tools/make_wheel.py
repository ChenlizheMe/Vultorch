"""Package a wheel from the compiled _vultorch module.

Called automatically by CMake after building _vultorch.
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
    """Find the _vultorch extension module and return (path, tag)."""
    for f in VULTORCH_DIR.iterdir():
        if f.name.startswith("_vultorch") and f.suffix in (".pyd", ".so"):
            # _vultorch.cp39-win_amd64.pyd  →  cp39-win_amd64
            parts = f.stem.split(".", 1)
            if len(parts) == 2:
                platform_tag = parts[1]   # e.g. cp39-win_amd64
                segs = platform_tag.split("-")
                if len(segs) == 2:
                    tag = f"{segs[0]}-{segs[0]}-{segs[1]}"
                elif len(segs) >= 3:
                    tag = platform_tag
                else:
                    tag = platform_tag
                return f, tag
    raise RuntimeError("No _vultorch.*.pyd/.so found in vultorch/")


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
        for f in sorted(VULTORCH_DIR.iterdir()):
            if f.name.startswith("__pycache__"):
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
            "Requires-Python: >=3.9\n"
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
