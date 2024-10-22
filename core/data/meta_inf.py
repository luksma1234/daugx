from daugx.core.augmentation.annotations import Annotations


class MetaInf:
    def __init__(self, annotations: Annotations):
        """
        Collection of all available meta information methods.
        Meta information is used for filtering or analysis.
        """
        self.__annotations = annotations
        self.__img_width = annotations.width
        self.__img_height = annotations.height
        self.__annotation_label_ids = [annotation.label.id for annotation in annotations.annots]
        self.__annotation_label_names = [annotation.label.name for annotation in annotations.annots]
        self.__annotation_max_area = max(
            [annotation.area for annotation in annotations.annots]
        ) if annotations.annots else None
        self.__annotation_min_area = min(
            [annotation.area for annotation in annotations.annots]
        ) if annotations.annots else None

    @property
    def annotations(self):
        return self.__annotations

    @property
    def label_ids(self):
        return [annotation.label.id for annotation in self.__annotations.annots]

    @property
    def label_names(self):
        return [annotation.label.name for annotation in self.__annotations.annots]

    @property
    def n_annotations(self):
        return len(self.__annotations.annots)

    @property
    def min_area(self):
        return min([annotation.area for annotation in self.__annotations.annots])

    @property
    def max_area(self):
        return max([annotation.area for annotation in self.__annotations.annots])

    @property
    def min_width(self):
        return min([annotation.width for annotation in self.__annotations.annots])

    @property
    def max_width(self):
        return max([annotation.width for annotation in self.__annotations.annots])

    @property
    def min_height(self):
        return min([annotation.height for annotation in self.__annotations.annots])

    @property
    def max_height(self):
        return max([annotation.height for annotation in self.__annotations.annots])

    def min_area_by_label_name(self, label_name):
        annotation_areas = [
            annotation.area for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_areas:
            return min(annotation_areas)
        return None

    def min_area_by_label_id(self, label_id):
        annotation_areas = [
            annotation.area for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_areas:
            return min(annotation_areas)
        return None

    def max_area_by_label_name(self, label_name):
        annotation_areas = [
            annotation.area for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_areas:
            return max(annotation_areas)
        return None

    def max_area_by_label_id(self, label_id):
        annotation_areas = [
            annotation.area for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_areas:
            return max(annotation_areas)
        return None

    def min_width_by_label_name(self, label_name):
        annotation_widths = [
            annotation.width for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_widths:
            return min(annotation_widths)
        return None

    def min_width_by_label_id(self, label_id):
        annotation_widths = [
            annotation.width for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_widths:
            return min(annotation_widths)
        return None

    def max_width_by_label_name(self, label_name):
        annotation_widths = [
            annotation.width for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_widths:
            return max(annotation_widths)
        return None

    def max_width_by_label_id(self, label_id):
        annotation_widths = [
            annotation.width for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_widths:
            return max(annotation_widths)
        return None

    def min_height_by_label_name(self, label_name):
        annotation_heights = [
            annotation.height for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_heights:
            return min(annotation_heights)
        return None

    def min_height_by_label_id(self, label_id):
        annotation_heights = [
            annotation.height for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_heights:
            return min(annotation_heights)
        return None

    def max_height_by_label_name(self, label_name):
        annotation_heights = [
            annotation.height for annotation in self.__annotations.annots if annotation.label.name == label_name
        ]
        if annotation_heights:
            return max(annotation_heights)
        return None

    def max_height_by_label_id(self, label_id):
        annotation_heights = [
            annotation.height for annotation in self.__annotations.annots if annotation.label.id == label_id
        ]
        if annotation_heights:
            return max(annotation_heights)
        return None

    def n_annotations_by_label_name(self, label_name):
        annotations = [annotation for annotation in self.__annotations if annotation.label.name == label_name]
        return len(annotations)

    def n_annotations_by_label_id(self, label_id):
        annotations = [annotation for annotation in self.__annotations if annotation.label.id == label_id]
        return len(annotations)

