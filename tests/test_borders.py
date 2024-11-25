import pytest

from daugx.core.augmentation.borders import ImageBorder


@pytest.fixture
def border():
    return ImageBorder(100, 100)

def test_width(border):
    assert border.width == 100

def test_height(border):
    assert border.height == 100

def test_x_min(border):
    assert border.x_min == 0

def test_x_max(border):
    assert border.x_max == 100

def test_y_min(border):
    assert border.y_min == 0

def test_y_max(border):
    assert border.y_max == 100

def test_corners(border):
    corners = border.corners
    assert corners[0,0] == 0
    assert corners[0,1] == 0
    assert corners[1,0] == 100
    assert corners[1,1] == 100

def test_area(border):
    assert border.area == 10000

def test_set(border):
    border.set(50, 50)
    assert border.x_min == 50
    assert border.y_min == 50
    assert border.x_max == 100
    assert border.y_max == 100

def test_reset(border):
    border.set(50, 50)
    border.reset()
    assert border.x_min == 0
    assert border.y_min == 0
    assert border.x_max == 100
    assert border.y_max == 100

def test_rebase(border):
    border.set(50, 50)
    border.rebase()
    assert border.x_min == 0
    assert border.y_min == 0
    assert border.x_max == 50
    assert border.y_max == 50

def test_scale(border):
    border.scale(1.5, 4)
    assert border.x_min == 0
    assert border.y_min == 0
    assert border.x_max == 150
    assert border.y_max == 400
