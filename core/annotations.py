import numpy as np
from typing import Tuple, List
from ..utils import new_id
from ..utils.mat_utils import get_2d_transf_mat


# TODO: Implement 0 to 1 range for annotations
# TODO: Rewrite the cut_min / cut_max function
# TODO: Implement the Affine transformation function for an image and for the annotations


class Label:
    def __init__(self, label_id: int, name: str = None):
        self.id = label_id
        self.name = name


class Annotation:
    def __init__(self, boundary: np.ndarray, label: Label):
        """
        A Roof class for all annotations.
        :param boundary: Any boundary of any object as numpy array. Can be a mask, a box, etc...
                         A boundary always includes two or more two-dimensional points on the image (x, y) plane.
                         Boundaries are always normalized and will therefore be forced in the range of 0 to 1.
        :param label: Any form of label represented as a dict.
        """
        self.boundary = boundary
        self.label = label
        self.id = new_id()
        self.size = self._get_size()
        self.pos = self._get_pos()
        self.invalid = False
        self.verify()

    def set_label(self, label: Label):
        self.label = label

    def verify(self):
        """
        Clips boundary to (0, 1) and verifies if boundary is still in image.
        Flags annotation as invalid if all matrix entries are equally 1 or 0.
        """
        self.boundary = np.clip(self.boundary, 0, 1)
        if np.all(self.boundary == 0) or np.all(self.boundary == 1):
            self.invalid = True

    def _get_size(self) -> int:
        """
        OVERRIDE IN SUBCLASS
        """
        pass

    def _get_pos(self) -> Tuple[float, float]:
        """
        Position is always the tuple (x_min, y_min) for all x and y values in boundary.
        """
        t_boundary = np.transpose(self.boundary)
        return min(t_boundary[0]), min(t_boundary[1])

    def set_boundary(self, boundary):
        self.__init__(boundary, self.label)


class Annotations:
    def __init__(self, img_dims: Tuple[int, int]):
        self.annots: List[Annotation] = []

    def affine(self, transf_matrix):
        self.transf(transf_matrix)

    def add(self, annot: Annotation):
        self.annots.append(annot)

    def filter(
            self,
            min_size: int = None,
            max_size: int = None,
            max_pos: Tuple[int, int] = None,
            min_pos: Tuple[int, int] = None,
            label: Label = None
    ):
        filter_annots = []
        for annot in self.annots:
            # check min size
            if min_size is not None and annot.size < min_size:
                filter_annots.append(annot)
            # check max size
            elif max_size is not None and annot.size > max_size:
                filter_annots.append(annot)
            # check max_pos
            elif (max_pos is not None
                  and any((dim_pos > dim_max_pos for dim_pos, dim_max_pos in zip(annot.pos, max_pos)))):
                filter_annots.append(annot)
            # check min_pos
            elif (min_pos is not None
                  and any((dim_pos < dim_min_pos for dim_pos, dim_min_pos in zip(annot.pos, min_pos)))):
                filter_annots.append(annot)
            elif label is not None and label.id == annot.label.id:
                filter_annots.append(annot)
        for annot in filter_annots:
            self.remove(annot)

    def remove(self, value):
        self.annots.remove(value)

    def transf(self, transf_mat: np.ndarray):
        """
        Applies a transformation matrix to all annotations.
        Uses einsum with batch index b, coordinate length i and j for coordinate depth.
        Coordinates are extended by 1 to fit the matrix size. After the transformation, the first two values of the
        resulting vector, are multiplied with the entry at index 2 (or 3 mathematically). This index performs scaling.
        Einsum returns vector with dims bx3. Unscales boundaries and reinitializes annotations with new boundaries.
        :param transf_mat: 3x3 Transformation Matrix
        """
        for annot in self.annots:
            boundary = annot.boundary
            exp_boundary = np.hstack((boundary, np.ones((len(boundary), 1))))
            transf_scaled_points = np.einsum("bi, ij -> bj", exp_boundary, transf_mat)
            annot.set_boundary(self._unscale(transf_scaled_points))

    @staticmethod
    def _unscale(scaled_points: np.ndarray):
        points = scaled_points[:, :2]
        scaler = np.vstack((scaled_points[:, 2], scaled_points[:, 2]))
        return np.einsum("ij, ji -> ij", points, scaler)


class BoxAnnotation(Annotation):
    def __init__(self, boundary: np.ndarray, label: Label):
        self._assure_box()
        self.width = self.boundary[2] - self.boundary[0]
        self.height = self.boundary[3] - self.boundary[1]
        super().__init__(boundary, label)

    def _get_size(self) -> int:
        return int(self.width * self.height)

    def _assure_box(self):
        """
        Sets boundary so that the box shape with [[x_min, y_min], [x_max, y_max]] is assured.
        """
        t_boundary = np.transpose(self.boundary)
        self.boundary = np.array(
            [
                [min(t_boundary[0]), min(t_boundary[1])],
                [max(t_boundary[0]), max(t_boundary[1])]
            ]
        )









































