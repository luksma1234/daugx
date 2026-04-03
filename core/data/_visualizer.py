"""Visualization helpers for Sample.show()."""
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np

from daugx.core.data.component import Component
from daugx.core.data.components.bounding_box import (
    ImageBoundingBox,
)
from daugx.core.data.components.image import Image
from daugx.core.data.components.keypoint import ImageKeyPoint
from daugx.core.data.components.polygon import ImagePolygon

_ALPHA = 0.3  # fill opacity
_OUTLINE_THICKNESS = 1
_KEYPOINT_RADIUS = 4
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.4
_FONT_THICKNESS = 1
_SPATIAL_ANNOTATION_TYPES = (
    ImageBoundingBox, ImagePolygon, ImageKeyPoint,
)


def _label_color(key) -> Tuple[int, int, int]:
    """Return a reproducible BGR color for a label key.

    Args:
        key: ``class_id`` (int) or ``class_name`` (str) or
            any hashable value.

    Returns:
        ``(B, G, R)`` tuple with each channel in [50, 220].
    """
    seed = hash(key) % (2 ** 32)
    rng = np.random.default_rng(seed)
    b, g, r = (int(x) for x in rng.integers(50, 220, size=3))
    return (b, g, r)


def _annotation_color(comp: Component) -> Tuple[int, int, int]:
    """Derive display color from a component's label."""
    key = getattr(comp, 'class_id', None)
    if key is None:
        key = getattr(comp, 'class_name', None)
    if key is None:
        key = 0
    return _label_color(key)


def _draw_bbox(
    canvas: np.ndarray,
    bbox: ImageBoundingBox,
    color: Tuple[int, int, int],
) -> None:
    """Draw a bounding box with transparent fill and opaque
    outline on *canvas*.

    Args:
        canvas: BGR image array (modified in-place).
        bbox: Bounding box annotation.
        color: BGR color tuple.
    """
    x0 = int(bbox.x_min)
    y0 = int(bbox.y_min)
    x1 = int(bbox.x_max)
    y1 = int(bbox.y_max)

    # Transparent fill via alpha blend
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(
        overlay, _ALPHA, canvas, 1 - _ALPHA, 0, canvas,
    )

    # Opaque outline
    cv2.rectangle(
        canvas, (x0, y0), (x1, y1), color, _OUTLINE_THICKNESS,
    )

    # Label
    label = _label_text(bbox)
    _draw_label(canvas, label, x0, max(y0 - 2, 0), color)


def _draw_polygon(
    canvas: np.ndarray,
    polygon: ImagePolygon,
    color: Tuple[int, int, int],
) -> None:
    """Draw a polygon with transparent fill and opaque outline.

    Args:
        canvas: BGR image array (modified in-place).
        polygon: Polygon annotation.
        color: BGR color tuple.
    """
    pts = polygon.points.astype(np.int32).reshape(-1, 1, 2)

    # Transparent fill
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(
        overlay, _ALPHA, canvas, 1 - _ALPHA, 0, canvas,
    )

    # Opaque outline
    cv2.polylines(
        canvas, [pts], True, color, _OUTLINE_THICKNESS,
    )

    # Label at centroid
    cx, cy = (int(v) for v in polygon.center)
    label = _label_text(polygon)
    _draw_label(canvas, label, cx, cy, color)


def _draw_keypoint(
    canvas: np.ndarray,
    kp: ImageKeyPoint,
    color: Tuple[int, int, int],
) -> None:
    """Draw a keypoint with transparent fill and opaque outline.

    Args:
        canvas: BGR image array (modified in-place).
        kp: Keypoint annotation.
        color: BGR color tuple.
    """
    cx = int(kp.x)
    cy = int(kp.y)

    # Transparent fill
    overlay = canvas.copy()
    cv2.circle(overlay, (cx, cy), _KEYPOINT_RADIUS, color, -1)
    cv2.addWeighted(
        overlay, _ALPHA, canvas, 1 - _ALPHA, 0, canvas,
    )

    # Opaque outline
    cv2.circle(
        canvas, (cx, cy), _KEYPOINT_RADIUS,
        color, _OUTLINE_THICKNESS,
    )

    # Label beside the point
    label = _label_text(kp)
    _draw_label(
        canvas, label, cx + _KEYPOINT_RADIUS + 2, cy, color,
    )


def _label_text(comp: Component) -> str:
    """Produce a display label from class_name or class_id."""
    name = getattr(comp, 'class_name', None)
    if name is not None:
        return str(name)
    cid = getattr(comp, 'class_id', None)
    if cid is not None:
        return str(cid)
    return ""


def _draw_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw a small text label at ``(x, y)``.

    A dark background is drawn first for legibility.

    Args:
        canvas: BGR image array (modified in-place).
        text: Label string.
        x: Left edge of the text.
        y: Bottom of the text (OpenCV convention).
        color: BGR color of the text.
    """
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(
        text, _FONT, _FONT_SCALE, _FONT_THICKNESS,
    )
    # Dark background rectangle
    cv2.rectangle(
        canvas,
        (x, y - th - 2),
        (x + tw + 2, y + 2),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        canvas, text, (x, y),
        _FONT, _FONT_SCALE, color, _FONT_THICKNESS,
        cv2.LINE_AA,
    )


def _draw_annotations(
    canvas: np.ndarray,
    annotations: List[Component],
) -> None:
    """Draw all supported spatial annotations onto *canvas*.

    Skips annotation types that have no spatial rendering
    (e.g. ``ImageCategory``).

    Args:
        canvas: BGR image array (modified in-place).
        annotations: Annotation components to draw.
    """
    for ann in annotations:
        color = _annotation_color(ann)
        if isinstance(ann, ImageBoundingBox):
            _draw_bbox(canvas, ann, color)
        elif isinstance(ann, ImagePolygon):
            _draw_polygon(canvas, ann, color)
        elif isinstance(ann, ImageKeyPoint):
            _draw_keypoint(canvas, ann, color)


def show_sample(
    sample,
    components: Optional[Tuple[Type[Component], ...]] = None,
) -> None:
    """Render and display the image(s) in *sample*.

    Called by ``Sample.show()``.

    Args:
        sample: The ``Sample`` to visualize.  Will be
            materialized if needed.
        components: Optional filter — only types listed
            here are included.

    Raises:
        ValueError: If the sample contains multiple
            ``Image`` components and *components* is
            ``None``.
    """
    sample.materialize()

    all_images = sample.get_all(Image)
    if not all_images:
        return

    #TODO: This does not make sense. There can be multiple image components to show, but all annotations must be
    # unambiguously assigned to one image. Therefore it has to be checked if all secondary components are assigned and
    # if all primary components can be shown together.
    # Also it must be possible to show Images by their own if needed. with Sample.show(daugx.Image)

    if components is None:
        if len(all_images) > 1:
            raise ValueError(
                f"Sample contains {len(all_images)} Image "
                f"components. Pass components= to specify "
                f"which to show (e.g. components=(Image,))."
            )
        images_to_show = all_images
        ann_filter = None  # show all spatial annotations
    else:
        images_to_show = [
            img for img in all_images
            if isinstance(img, components)
        ]
        ann_filter = tuple(
            t for t in components
            if t is not Image
        )

    for img in images_to_show:
        canvas = img.data.copy()

        # Gather annotations scoped to this image
        all_annotations = sample.get_annotations(
            target=img.name,
        )
        if ann_filter is not None:
            all_annotations = [
                a for a in all_annotations
                if isinstance(a, ann_filter)
            ]

        # Only draw spatial types
        spatial = [
            a for a in all_annotations
            if isinstance(a, _SPATIAL_ANNOTATION_TYPES)
        ]
        _draw_annotations(canvas, spatial)

        window_name = img.name if img.name else "Sample"
        cv2.imshow(window_name, canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
