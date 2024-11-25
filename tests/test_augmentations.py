"""
Tests for the annotations are mostly covered in test_annotations. This module focuses on the tests on the image.
Image coordinate basics:
    - Top left (0, 0)
    - downwards -> positive x
    - rightwards -> positive y
    - coordinates: (y, x, 3)
"""

import pytest
import numpy as np
from daugx.utils.misc import transpose_image
from daugx.core.augmentation.augmentations import Scale, Shift, Resize, MixUp, Mosaic, Crop, Rotate
from daugx.core.augmentation.annotations import Annotations
import cv2

@pytest.fixture
def annotations():
    annots = Annotations(
        image_width=100,
        image_height=100,
        boundary_type="BBoxBoundary",
        gen=np.random.default_rng(0)
    )
    annots.add(
        boundary_points=np.array(
            [[10, 10], [20, 20]]
        ),
        label_id=1,
        label_name="test"
    )
    return annots

@pytest.fixture
def grey_image():
    return np.ones((100, 100, 3), dtype=np.uint8) * 127

@pytest.fixture
def striped_grey_image_horizontal(grey_image):
    # white stripe at x = 50
    grey_image[50, :, :] = np.ones((1, 100, 3)) * 255
    return grey_image

@pytest.fixture
def striped_grey_image_vertical(grey_image):
    # white stripe at y = 50
    grey_image[:, 50, :] = np.ones((1, 100, 3)) * 255
    return grey_image

# Tests for Shift augmentation

@pytest.fixture
def positive_shift():
    return Shift(10, 15)

@pytest.fixture
def negative_shift():
    return Shift(-10, -15)

def test_shift_grey_positive_vertical(positive_shift, striped_grey_image_vertical):
    """
    Annots have already been tested. Shifts upwards to the right hand side.
    """
    image, annots = positive_shift.apply(striped_grey_image_vertical, None)
    # prev stipe x pos
    assert image[0, 50, 0] == 127
    # new stripe x pos
    assert image[0, 60, 0] == 255
    # new stripe ymax pos
    assert image[84, 60, 0] == 255
    # prev stripe ymax pos, background color
    assert image[99, 60, 0] == 0

def test_shift_grey_negative_horizontal(negative_shift, striped_grey_image_horizontal):
    """
    Annots have already been tested. Shifts downwards to the left hand side.
    """
    image, annots = negative_shift.apply(striped_grey_image_horizontal, None)
    # prev stripe y pos
    assert image[50, 0, 0] == 127
    # new stripe y pos
    assert image[65, 0, 0] == 255
    # new stripe xmin pos
    assert image[65, 84, 0] == 255
    # prev stripe xmax pos, background color
    assert image[65, 99, 0] == 0

# Tests for Scale augmentation

...

# Tests for Rotate augmentation

...

# Tests for Resize augmentation

...

# Tests for Mosaic augmentation

...

# Tests for Crop augmentation

...

# Tests for MixUp augmentation

...





















































