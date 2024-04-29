import numpy as np
from typing import Tuple, List
from ..utils import new_id


class Label:
    def __init__(self, label_id: int, name: str = None):
        self.id = label_id
        self.name = name


class Annotation:
    def __init__(self, boundary: np.ndarray, label: Label, reflect: bool = True):
        """
        A Roof class for all annotations.
        :param boundary: Any boundary of any object as numpy array. Can be a mask, a box, etc...
                         A boundary always includes two or more two-dimensional points on the image (x, y) plane.
        :param label: Any form of label represented as a dict.
        """
        self.boundary = boundary
        self.label = label
        self.id = new_id()
        self.size = self._get_size()
        self.pos = self._get_pos()
        self.invalid = False

    def set_label(self, label: Label):
        self.label = label

    def verify(self):
        """
        OVERRIDE IN SUBCLASS
        """
        pass

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

    def cut_min(self, new_min: Tuple[int, int]):
        """
        Cuts all points of boundary to a new min value.
        """
        # check if all points are outside new min
        if all(any(dim_new_min > dim_point for dim_new_min, dim_point in zip(new_min, point))
               for point in self.boundary):
            self.invalid = True
        else:
            self.boundary = np.array(
                [(dim_new_min if dim_new_min > dim_point else dim_point
                  for dim_new_min, dim_point in zip(new_min, point))
                 for point in self.boundary]
            )

    def cut_max(self, new_max: Tuple[int, int]):
        """
        Cuts all points of boundary to a new max_value.
        """
        # check if all points are outside new max
        if all(any(dim_new_max < dim_point for dim_new_max, dim_point in zip(new_max, point))
               for point in self.boundary):
            self.invalid = True
        else:
            self.boundary = np.array(
                [(dim_new_max if dim_new_max < dim_point else dim_point
                  for dim_new_max, dim_point in zip(new_max, point))
                 for point in self.boundary]
            )

    def check_size(self, min_size: int = None, max_size: int = None):
        if min_size is not None:
            self.invalid = self.size < min_size
        if max_size is not None:
            if not self.invalid:
                self.invalid = self.size > max_size


# TODO: Make annotations a decorator for each augmentation which affects annotations

class Annotations:
    def __init__(self, img_dims: Tuple[int, int]):
        self.annots: List[Annotation] = []
        self.img_dims = img_dims

    def shift(self, shift: Tuple[float, float]):
        self.transf(self.get_transf_mat(translation=shift))

    def scale(self, scale: Tuple[float, float]):
        self.transf(self.get_transf_mat(scale=scale))

    def rotate(self, angle: int or float):
        self.transf(self.get_transf_mat(angle=angle))

    def perspective(self, distortion: Tuple[float, float]):
        self.transf(self.get_transf_mat(distortion=distortion))

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
        scaler = np.hstack((scaled_points[:, 2], scaled_points[:, 2]))
        return np.einsum("ij, ji -> ij", points, scaler)

    @staticmethod
    def get_transf_mat(
            scale: Tuple[float, float] = None,
            angle: float = None,
            translation: Tuple[float, float] = None,
            distortion: Tuple[float, float] = None
    ):
        """
        Create a transformation matrix based on specified parameters.

        Args:
            scale (tuple): Scaling factors along x and y axes (sx, sy).
            angle (float): Rotation angle in degrees.
            translation (tuple): Translation along x and y axes (tx, ty).
            distortion (tuple): Perspective distortion coefficients (px, py).

        Returns:
            numpy.ndarray: Transformation matrix.
        """
        matrix = np.identity(3)

        if scale is not None:
            sx, sy = scale
            scale_matrix = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
            matrix = np.dot(scale_matrix, matrix)

        if angle is not None:
            angle_rad = np.radians(angle)
            rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                        [np.sin(angle_rad), np.cos(angle_rad), 0],
                                        [0, 0, 1]])
            matrix = np.dot(rotation_matrix, matrix)

        if translation is not None:
            tx, ty = translation
            translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
            matrix = np.dot(translation_matrix, matrix)

        if distortion is not None:
            dx, dy = distortion
            perspective_matrix = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
            matrix = np.dot(perspective_matrix, matrix)

        return matrix


class BoxAnnotation(Annotation):
    def __init__(self, boundary: np.ndarray, label: Label):
        self._permute_boundary()
        self.width = self.boundary[2] - self.boundary[0]
        self.height = self.boundary[3] - self.boundary[1]
        super().__init__(boundary, label)

    def verify(self):
        assert self.width >= 0 and self.height >= 0, \
            (f"Invalid annotation boundaries. Expected box width and height > 0. "
             f"Found (width, height): ({self.width, self.height}).")

    def _get_size(self) -> int:
        return int(self.width * self.height)

    def _permute_boundary(self):
        """
        Permutes boundary so that the box shape with [[x_min, y_min], [x_max, y_max]] is assured.
        """
        t_boundary = np.transpose(self.boundary)
        self.boundary = np.array(
            [
                [min(t_boundary[0]), min(t_boundary[1])],
                [max(t_boundary[0]), max(t_boundary[1])]
            ]
        )









































