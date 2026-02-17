"""Tests for type stubs and script file structure.

Coverage targets:
  - vultorch/__init__.pyi type stub consistency
  - vultorch/ui.pyi type stub consistency
  - vultorch/py.typed marker exists
  - scripts/setup_wsl2.sh is valid bash
  - pytest.ini configuration
  - pyproject.toml structure
"""

import pytest
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestTypeStubs:

    def test_init_pyi_exists(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        assert pyi.exists(), "__init__.pyi not found"

    def test_init_pyi_parseable(self):
        """The .pyi file should be valid Python syntax."""
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        ast.parse(content)  # raises SyntaxError if invalid

    def test_init_pyi_has_window(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "class Window" in content

    def test_init_pyi_has_show(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "def show" in content

    def test_init_pyi_has_create_tensor(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "def create_tensor" in content

    def test_init_pyi_has_camera_light(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "class Camera" in content
        assert "class Light" in content

    def test_init_pyi_has_scene_view(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "SceneView" in content

    def test_init_pyi_has_view_panel_canvas(self):
        pyi = ROOT / "vultorch" / "__init__.pyi"
        content = pyi.read_text(encoding="utf-8")
        assert "class View" in content
        assert "class Panel" in content
        assert "class Canvas" in content

    def test_ui_pyi_exists(self):
        pyi = ROOT / "vultorch" / "ui.pyi"
        assert pyi.exists(), "ui.pyi not found"

    def test_ui_pyi_parseable(self):
        pyi = ROOT / "vultorch" / "ui.pyi"
        content = pyi.read_text(encoding="utf-8")
        ast.parse(content)

    def test_ui_pyi_has_core_functions(self):
        pyi = ROOT / "vultorch" / "ui.pyi"
        content = pyi.read_text(encoding="utf-8")
        for fn in ["def text", "def button", "def begin", "def end",
                    "def slider_float", "def combo", "def image"]:
            assert fn in content, f"ui.pyi missing {fn}"

    def test_py_typed_marker(self):
        marker = ROOT / "vultorch" / "py.typed"
        assert marker.exists(), "py.typed marker missing (PEP 561)"


class TestProjectConfiguration:

    def test_pyproject_toml_exists(self):
        assert (ROOT / "pyproject.toml").exists()

    def test_pyproject_has_version(self):
        content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'version' in content

    def test_pyproject_has_project_name(self):
        content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert 'vultorch' in content

    def test_pytest_ini_exists(self):
        assert (ROOT / "pytest.ini").exists()

    def test_pytest_ini_markers(self):
        content = (ROOT / "pytest.ini").read_text(encoding="utf-8")
        assert "gpu" in content
        assert "slow" in content

    def test_pytest_ini_norecursedirs(self):
        content = (ROOT / "pytest.ini").read_text(encoding="utf-8")
        assert "external" in content

    def test_cmake_lists_exists(self):
        assert (ROOT / "CMakeLists.txt").exists()

    def test_mkdocs_yml_exists(self):
        assert (ROOT / "mkdocs.yml").exists()

    def test_license_exists(self):
        assert (ROOT / "LICENSE").exists()


class TestSetupScript:

    def test_setup_wsl2_exists(self):
        script = ROOT / "scripts" / "setup_wsl2.sh"
        assert script.exists()

    def test_setup_wsl2_has_shebang(self):
        script = ROOT / "scripts" / "setup_wsl2.sh"
        first_line = script.read_text(encoding="utf-8").split("\n")[0]
        assert first_line.startswith("#!")

    def test_setup_wsl2_has_set_e(self):
        """Script should use set -e for fail-fast."""
        content = (ROOT / "scripts" / "setup_wsl2.sh").read_text(encoding="utf-8")
        assert "set -e" in content

    def test_setup_wsl2_installs_cmake(self):
        content = (ROOT / "scripts" / "setup_wsl2.sh").read_text(encoding="utf-8")
        assert "cmake" in content

    def test_setup_wsl2_installs_vulkan(self):
        content = (ROOT / "scripts" / "setup_wsl2.sh").read_text(encoding="utf-8")
        assert "vulkan" in content.lower()


class TestExamplesExist:
    """Verify example files exist and are valid Python."""

    @pytest.mark.parametrize("name", [
        "01_hello_tensor.py",
        "02_imgui_controls.py",
        "03_training_test.py",
    ])
    def test_example_exists(self, name):
        assert (ROOT / "examples" / name).exists()

    @pytest.mark.parametrize("name", [
        "01_hello_tensor.py",
        "02_imgui_controls.py",
        "03_training_test.py",
    ])
    def test_example_parseable(self, name):
        content = (ROOT / "examples" / name).read_text(encoding="utf-8")
        ast.parse(content)


class TestTutorialsExist:
    """Verify tutorial markdown files exist."""

    @pytest.mark.parametrize("name", [
        "index.md", "index.zh.md",
        "01_hello_tensor.md", "01_hello_tensor.zh.md",
        "02_multi_panel.md", "02_multi_panel.zh.md",
        "03_training_test.md", "03_training_test.zh.md",
    ])
    def test_tutorial_exists(self, name):
        assert (ROOT / "tutorial" / name).exists()
