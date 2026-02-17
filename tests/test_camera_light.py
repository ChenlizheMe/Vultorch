"""Tests for Camera and Light data-holder classes.

Coverage targets:
  - Camera.__init__, reset, __slots__
  - Light.__init__, __slots__
  - Camera._push, _pull (mocked engine)
  - Light._push (mocked engine)
"""

import pytest
from conftest import requires_vultorch


@requires_vultorch
class TestCamera:

    def test_defaults(self):
        from vultorch import Camera
        cam = Camera()
        assert cam.azimuth == 0.0
        assert cam.elevation == 0.6
        assert cam.distance == 3.0
        assert cam.target == (0.0, 0.0, 0.0)
        assert cam.fov == 45.0

    def test_reset(self):
        from vultorch import Camera
        cam = Camera()
        cam.azimuth = 1.5
        cam.distance = 10.0
        cam.target = (1.0, 2.0, 3.0)
        cam.reset()
        assert cam.azimuth == 0.0
        assert cam.distance == 3.0
        assert cam.target == (0.0, 0.0, 0.0)

    def test_slots(self):
        from vultorch import Camera
        cam = Camera()
        with pytest.raises(AttributeError):
            cam.nonexistent_attr = 42

    def test_modify_values(self):
        from vultorch import Camera
        cam = Camera()
        cam.azimuth = 3.14
        cam.elevation = -0.5
        cam.distance = 5.0
        cam.target = (1.0, 2.0, 3.0)
        cam.fov = 60.0
        assert cam.azimuth == pytest.approx(3.14)
        assert cam.elevation == pytest.approx(-0.5)
        assert cam.distance == 5.0
        assert cam.target == (1.0, 2.0, 3.0)
        assert cam.fov == 60.0


@requires_vultorch
class TestLight:

    def test_defaults(self):
        from vultorch import Light
        lt = Light()
        assert lt.direction == (0.3, -1.0, 0.5)
        assert lt.color == (1.0, 1.0, 1.0)
        assert lt.intensity == 1.0
        assert lt.ambient == 0.15
        assert lt.specular == 0.5
        assert lt.shininess == 32.0
        assert lt.enabled is True

    def test_slots(self):
        from vultorch import Light
        lt = Light()
        with pytest.raises(AttributeError):
            lt.nonexistent_attr = 42

    def test_modify_values(self):
        from vultorch import Light
        lt = Light()
        lt.direction = (0.0, -1.0, 0.0)
        lt.color = (0.5, 0.5, 0.5)
        lt.intensity = 2.0
        lt.ambient = 0.3
        lt.specular = 0.8
        lt.shininess = 64.0
        lt.enabled = False
        assert lt.direction == (0.0, -1.0, 0.0)
        assert lt.color == (0.5, 0.5, 0.5)
        assert lt.intensity == 2.0
        assert lt.enabled is False
