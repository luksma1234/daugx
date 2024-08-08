import numpy as np
from typing import Tuple


def get_2d_transf_mat(
        scale: Tuple[float, float] = None,
        angle: float = None,
        translation: Tuple[float, float] = None,
        shear: Tuple[float, float] = None
):
    """
    Create a transformation matrix based on specified parameters.

    Args:
        scale (tuple): Scaling factors along x and y axes (sx, sy).
        angle (float): Rotation angle in degrees.
        translation (tuple): Translation along x and y axes (tx, ty).
        shear (tuple): Perspective distortion coefficients (px, py).

    Returns:
        numpy.ndarray: Transformation matrix.
    """
    matrix = np.identity(3)

    if scale is not None:
        sx, sy = scale
        scale_matrix = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1]
        ])
        matrix = np.dot(scale_matrix, matrix)

    if angle is not None:
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
        matrix = np.dot(rotation_matrix, matrix)

    if translation is not None:
        tx, ty = translation
        translation_matrix = np.array([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])
        matrix = np.dot(translation_matrix, matrix)

    if shear is not None:
        dx, dy = shear
        perspective_matrix = np.array([
            [1, dx, 0],
            [dy, 1, 0],
            [0, 0, 1]
        ])
        matrix = np.dot(perspective_matrix, matrix)

    return matrix


def img_to_mat_repr():
    pass


def mat_to_im_repr():
    pass