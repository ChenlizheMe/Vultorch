"""Tests for tools/make_wheel.py — wheel packaging utility.

Coverage targets:
  - get_version(): correctly parses pyproject.toml
  - _record_line(): produces correct sha256 + size RECORD entries
  - find_pyd(): error when no extension module present
  - make_wheel(): full packaging (when .pyd/.so exists)
"""

import hashlib
import base64
import importlib
import pytest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent


def _load_make_wheel():
    """Import tools/make_wheel.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "make_wheel", ROOT / "tools" / "make_wheel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestGetVersion:

    def test_returns_string(self):
        mod = _load_make_wheel()
        ver = mod.get_version()
        assert isinstance(ver, str)
        assert len(ver) > 0

    def test_semver_like(self):
        mod = _load_make_wheel()
        ver = mod.get_version()
        parts = ver.split(".")
        assert len(parts) >= 2, f"Version '{ver}' should have at least major.minor"

    def test_matches_pyproject(self):
        """Version must match what is in pyproject.toml."""
        import re
        mod = _load_make_wheel()
        ver = mod.get_version()
        toml = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'version\s*=\s*"([^"]+)"', toml)
        assert m is not None
        assert ver == m.group(1)

    def test_missing_version_raises(self, tmp_path):
        """If pyproject.toml has no version line, raise RuntimeError."""
        mod = _load_make_wheel()
        fake_toml = tmp_path / "pyproject.toml"
        fake_toml.write_text("[project]\nname = 'test'\n")
        with patch.object(mod, "ROOT", tmp_path):
            with pytest.raises(RuntimeError, match="Cannot find version"):
                mod.get_version()


class TestRecordLine:

    def test_format(self):
        mod = _load_make_wheel()
        data = b"hello world"
        line = mod._record_line("vultorch/test.py", data)
        parts = line.split(",")
        assert len(parts) == 3
        assert parts[0] == "vultorch/test.py"
        assert parts[1].startswith("sha256=")
        assert parts[2] == str(len(data))

    def test_sha256_correct(self):
        mod = _load_make_wheel()
        data = b"test data for hashing"
        line = mod._record_line("arc", data)
        sha_part = line.split(",")[1]
        b64_hash = sha_part.replace("sha256=", "")
        # Verify by recomputing
        digest = hashlib.sha256(data).digest()
        expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        assert b64_hash == expected

    def test_empty_data(self):
        mod = _load_make_wheel()
        line = mod._record_line("empty.txt", b"")
        parts = line.split(",")
        assert parts[2] == "0"

    def test_large_data(self):
        mod = _load_make_wheel()
        data = b"\x00" * 100000
        line = mod._record_line("big.bin", data)
        parts = line.split(",")
        assert parts[2] == "100000"


class TestFindPyd:

    def test_no_extension_raises(self, tmp_path):
        """find_pyd should raise if no _vultorch.*.pyd/.so exists."""
        mod = _load_make_wheel()
        empty_pkg = tmp_path / "vultorch"
        empty_pkg.mkdir()
        (empty_pkg / "__init__.py").write_text("")
        with patch.object(mod, "VULTORCH_DIR", empty_pkg):
            with pytest.raises(RuntimeError, match="No _vultorch"):
                mod.find_pyd()

    def test_finds_existing_extension(self):
        """If a real extension module exists, find_pyd should return it."""
        mod = _load_make_wheel()
        # Check if there actually is an extension to find
        exts = list(mod.VULTORCH_DIR.glob("_vultorch*"))
        if not exts:
            pytest.skip("No compiled _vultorch extension available")
        path, tag = mod.find_pyd()
        assert path.exists()
        assert isinstance(tag, str)
        assert len(tag) > 0


class TestMakeWheel:

    def test_make_wheel_produces_file(self):
        """If extension module exists, make_wheel should produce a .whl."""
        mod = _load_make_wheel()
        exts = list(mod.VULTORCH_DIR.glob("_vultorch*"))
        if not exts:
            pytest.skip("No compiled _vultorch extension available")
        whl_path = mod.make_wheel()
        assert whl_path.exists()
        assert whl_path.suffix == ".whl"

    def test_wheel_contents(self):
        """The wheel should contain METADATA, WHEEL, RECORD, and package files."""
        import zipfile
        mod = _load_make_wheel()
        exts = list(mod.VULTORCH_DIR.glob("_vultorch*"))
        if not exts:
            pytest.skip("No compiled _vultorch extension available")
        whl_path = mod.make_wheel()
        with zipfile.ZipFile(whl_path, "r") as zf:
            names = zf.namelist()
            assert any("METADATA" in n for n in names)
            assert any("WHEEL" in n for n in names)
            assert any("RECORD" in n for n in names)
            assert any("__init__.py" in n for n in names)
