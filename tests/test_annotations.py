import pytest
import numpy as np


from daugx.core.augmentation.annotations import Annotations
from daugx.core.augmentation.annotations import Annotation
from daugx.core.augmentation.annotations import Label
from daugx.core.augmentation.borders import ImageBorder

# Tests for Label class ################################################################################################
@pytest.fixture
def label_with_id():
    return Label(label_id=1)

@pytest.fixture
def label_with_name():
    return Label(name="test")

@pytest.fixture
def label_with_id_and_name():
    return Label(label_id=1, name="test")

def test_empty_label():
    try:
        _ = Label()
    except AssertionError:
        return

def test_label_name(label_with_name):
    assert label_with_name.name is not None
    assert label_with_name.id is None
    assert isinstance(label_with_name.name, str)

def test_label_id(label_with_id):
    assert label_with_id.id is not None
    assert label_with_id.name is None
    assert isinstance(label_with_id.id, int)

def test_label_name_and_id(label_with_id_and_name):
    assert label_with_id_and_name.id is not None
    assert label_with_id_and_name.name is not None
    assert isinstance(label_with_id_and_name.id, int)
    assert  isinstance(label_with_id_and_name.name, str)

# Tests for Annotation class ###########################################################################################

@pytest.fixture
def annotation():
    return Annotation(
        boundary_points=np.array(
        [[10, 10], [20, 20]]
        ),
        image_border=ImageBorder(100, 100),
        boundary_type="BBoxBoundary",
        uuid="abc",
        label_id=1,
        label_name="test"
    )

def test_annotation_boundary(annotation):
    points = annotation.boundary.points
    assert points[0, 0] == 10
    assert points[0, 1] == 10
    assert points[1, 0] == 20
    assert points[1, 1] == 20

def test_annotation_label(annotation):
    assert annotation.label.name == "test"
    assert annotation.label.id == 1

# Tests for Annotations class ##########################################################################################

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

def test_annotations_annots(annotations):
    assert len(annotations.annots) == 1
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 10
    assert annot_points[0, 1] == 10
    assert annot_points[1, 0] == 20
    assert annot_points[1, 1] == 20
    assert annot.label.name == "test"
    assert annot.label.id == 1

def test_annotations_border(annotations):
    assert annotations.width == 100
    assert annotations.height == 100

def test_annotations_clean(annotations):
    annotations.annots[0].valid = False
    annotations.clean()
    assert len(annotations.annots) == 0

def test_annotations_set_border(annotations):
    annotations.set_border(10, 10, 50, 50)
    assert annotations.width == 40
    assert annotations.height == 40

def test_scale_border(annotations):
    annotations.scale_border(2, 2)
    assert annotations.width == 200
    assert annotations.height == 200

def test_filter(annotations):
    annotations.add(
        boundary_points=np.array(
            [[50, 10], [70, 20]]
        ),
        label_id=2,
        label_name="test2"
    )
    annotations.filter(drop_labels=[1])
    assert len(annotations.annots) == 1
    assert annotations.annots[0].label.name == "test2"

def test_positive_shift(annotations):
    annotations.shift(10, 50)
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 20
    assert annot_points[0, 1] == 60
    assert annot_points[1, 0] == 30
    assert annot_points[1, 1] == 70

def test_positive_shift_out_of_border(annotations):
    annotations.shift(100, 50)
    # shifts boundary out of border
    assert len(annotations.annots) == 0

def test_negative_shift(annotations):
    annotations.shift(-10, -10)
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 0
    assert annot_points[0, 1] == 0
    assert annot_points[1, 0] == 10
    assert annot_points[1, 1] == 10

def test_negative_shift_out_of_border(annotations):
    annotations.shift(-10, -50)
    # shifts boundary out of border
    assert len(annotations.annots) == 0

def test_positive_scale(annotations):
    annotations.scale(1.5, 2.5)
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 15
    assert annot_points[0, 1] == 25
    assert annot_points[1, 0] == 30
    assert annot_points[1, 1] == 50
    assert annotations.width == 150
    assert annotations.height == 250

def test_negative_scale(annotations):
    annotations.scale(0.5, 0.25)
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 5
    assert annot_points[0, 1] == 2.5
    assert annot_points[1, 0] == 10
    assert annot_points[1, 1] == 5
    assert annotations.width == 50
    assert annotations.height == 25

def test_crop(annotations):
    annotations.crop(10, 5, 15, 100)
    annot = annotations.annots[0]
    annot_points = annot.boundary.points
    assert annot_points[0, 0] == 0
    assert annot_points[0, 1] == 5
    assert annot_points[1, 0] == 5
    assert annot_points[1, 1] == 15
    assert annotations.width == 5
    assert annotations.height == 95
