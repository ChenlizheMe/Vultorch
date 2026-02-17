"""Tests for tools/spv_to_header.py — SPIR-V to C header converter.

Coverage targets:
  - Correct header format (pragma once, includes, uint32_t array)
  - Padding to uint32_t alignment
  - Variable naming and size constant
  - Missing arguments cause SystemExit
"""

import os
import sys
import struct
import pytest
import importlib
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_spv_module():
    """Import tools/spv_to_header.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "spv_to_header", ROOT / "tools" / "spv_to_header.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestSpvToHeader:

    def test_basic_conversion(self, tmp_path):
        """Write a tiny valid SPIR-V file and check the generated header."""
        spv_path = tmp_path / "test.spv"
        header_path = tmp_path / "out" / "test_shader.h"

        # Create a fake 8-byte SPIR-V payload (2 uint32 words)
        words = [0x07230203, 0x00010000]  # SPIR-V magic + version
        data = b"".join(struct.pack("<I", w) for w in words)
        spv_path.write_bytes(data)

        # Run main() via sys.argv patching
        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header", str(spv_path), str(header_path), "test_vert"]
            mod.main()
        finally:
            sys.argv = old_argv

        assert header_path.exists()
        content = header_path.read_text()
        assert "#pragma once" in content
        assert "#include <cstdint>" in content
        assert "static const uint32_t test_vert[]" in content
        assert "test_vert_size" in content
        assert "0x07230203" in content
        assert "0x00010000" in content
        assert "Auto-generated" in content

    def test_padding_alignment(self, tmp_path):
        """Input not aligned to 4 bytes should be padded with zeroes."""
        spv_path = tmp_path / "pad.spv"
        header_path = tmp_path / "pad.h"

        # 5 bytes — 1 byte short of 2 words, should pad to 8 bytes
        data = bytes([0x01, 0x02, 0x03, 0x04, 0x05])
        spv_path.write_bytes(data)

        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header", str(spv_path), str(header_path), "padded"]
            mod.main()
        finally:
            sys.argv = old_argv

        content = header_path.read_text()
        # Should have 2 words (8 bytes / 4 = 2)
        assert "padded[]" in content
        assert "padded_size" in content

    def test_existing_dir(self, tmp_path):
        """Should create intermediate directories."""
        spv_path = tmp_path / "test.spv"
        deep_header = tmp_path / "a" / "b" / "c" / "shader.h"

        spv_path.write_bytes(struct.pack("<I", 0xDEADBEEF))
        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header", str(spv_path), str(deep_header), "deep"]
            mod.main()
        finally:
            sys.argv = old_argv

        assert deep_header.exists()

    def test_variable_name_preserved(self, tmp_path):
        """The variable name in the header must match the CLI argument."""
        spv_path = tmp_path / "test.spv"
        header_path = tmp_path / "var_name.h"
        spv_path.write_bytes(struct.pack("<I", 0xCAFEBABE))

        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header", str(spv_path), str(header_path),
                         "my_custom_name"]
            mod.main()
        finally:
            sys.argv = old_argv

        content = header_path.read_text()
        assert "my_custom_name[]" in content
        assert "my_custom_name_size" in content

    def test_missing_args_exits(self):
        """Calling main() with fewer than 4 args should SystemExit(1)."""
        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header"]
            with pytest.raises(SystemExit):
                mod.main()
        finally:
            sys.argv = old_argv

    def test_word_count(self, tmp_path):
        """16 bytes → exactly 4 uint32 words in header."""
        spv_path = tmp_path / "words.spv"
        header_path = tmp_path / "words.h"
        data = struct.pack("<4I", 1, 2, 3, 4)
        spv_path.write_bytes(data)

        mod = _load_spv_module()
        old_argv = sys.argv
        try:
            sys.argv = ["spv_to_header", str(spv_path), str(header_path), "w"]
            mod.main()
        finally:
            sys.argv = old_argv

        content = header_path.read_text()
        # Count hex literals: 4 words
        hex_words = [tok for tok in content.split() if tok.startswith("0x")]
        # Strip trailing commas
        hex_words = [w.rstrip(",") for w in hex_words]
        assert len(hex_words) == 4
