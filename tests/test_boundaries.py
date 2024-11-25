import pytest
import numpy as np

from daugx.core.augmentation.borders import ImageBorder
from daugx.core.augmentation.boundaries import Boundary, BBoxBoundary, PolyBoundary, KeyPBoundary


@pytest.fixture
def boundary():
    border = ImageBorder(100, 100)
    points = np.array(
        [[10, 10], [20, 20]]
    )
    return Boundary(points, border)

def test_points(boundary):
    points = boundary.points
    assert points[0,0] == 10
    assert points[0,1] == 10
    assert points[1,0] == 20
    assert points[1,1] == 20

def test_width(boundary):
    assert boundary.width == 10

def test_height(boundary):
    assert boundary.height == 10

def test_center(boundary):
    center = boundary.center
    assert center[0] == 15.0
    assert center[1] == 15.0

def test_set(boundary):
    points = np.array(
        [[2, 2], [49, 49]]
    )
    boundary.set(points, False)
    assert boundary.width == 47
    assert boundary.height == 47
    center = boundary.center
    assert center[0] == 25.5
    assert center[1] == 25.5

def test_clip(boundary):
    points = np.array(
        [[20, 20], [200, 200]]
    )
    boundary.set(points)
    boundary.clip()
    assert boundary.width == 80
    assert boundary.height == 80
    center = boundary.center
    assert center[0] == 60
    assert center[1] == 60

def test_shift(boundary):
    boundary.shift(10.5, 20.5)
    assert boundary.width == 10
    assert boundary.height == 10
    center = boundary.center
    assert center[0] == 25.5
    assert center[1] == 35.5

def test_scale(boundary):
    boundary.scale(2.5,  1.5, False)
    assert boundary.width == 25
    assert boundary.height == 15
    center = boundary.center
    assert center[0] == 37.5
    assert center[1] == 22.5

def test_rotate(boundary):
    boundary.rotate(90)
    assert np.round(boundary.width, 5) == 10.0
    assert np.round(boundary.height, 5) == 10.0
    center = boundary.center
    assert np.round(center[0], 5) == 85.0
    assert np.round(center[1], 5) == 15.0


@pytest.fixture
def bbox_boundary():
    border = ImageBorder(100, 100)
    points = np.array(
        [[10, 10], [20, 20], [15, 15]]
    )
    return BBoxBoundary(points, border)

def test_bbox_points(bbox_boundary):
    points = bbox_boundary.points
    assert np.shape(points) == (2, 2)
    assert points[0, 0] == 10
    assert points[0, 1] == 10
    assert points[1, 0] == 20
    assert points[1, 1] == 20

def test_bbox_area(bbox_boundary):
    assert bbox_boundary.area == 100

@pytest.fixture
def poly_boundary():
    border = ImageBorder(100, 100)
    points = np.array(
        [[10, 10], [10, 20], [20, 15]]
    )
    return PolyBoundary(points, border)

def test_poly_area(poly_boundary):
    assert poly_boundary.area == 50
