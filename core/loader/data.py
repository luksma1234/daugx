from copy import deepcopy
from operator import itemgetter
from typing import Tuple, List, Dict

from daugx.core.augmentation.annotations import Annotations, Label
from daugx.utils.misc import read_img
from daugx.core.loader.loader import AnnotationLoader

import numpy as np


class DataPackage:

    def __init__(self, image_path: str, image_dims: Tuple[int, int], raw_annotations: List[dict], annotation_type: str):
        """
        A high level wrapper to wrap annotations and image data. Remains read-only after initialization. Saves image
        and annotation meta information in initialization.
        Args:
            image_path (str): Absolute path to image
            image_dims (Tuple[int, int]): Image dimensions, ignoring channels
            raw_annotations (List[dict]): Raw annotation data from Loader. Data format:

              [
                {
                    label_name: (str)
                    label_id: (int)
                    boundary: (np.ndarray)
                },
                ...
              ]
        """
        self.__image_path = image_path
        self.__image_dims = image_dims
        self.__raw_annotations = raw_annotations
        self.__annotation_type = annotation_type
        self.__annotations = None

        self._annotations_from_dict()

    @property
    def meta_inf(self) -> dict:
        """
        Meta Information of image and its annotations. Meta information is used to filter Data Packages.
        Returns a dict of Meta Information.
        """
        return self._retrieve_meta_inf()

    @property
    def data(self) -> Tuple[np.ndarray, Annotations]:
        """
        Data Property. Returns a Tuple of the loaded image and its annotations.
        """
        return self._load_image(), deepcopy(self.__annotations)

    def _load_image(self):
        return read_img(self.__image_path)

    def _annotations_from_dict(self) -> None:
        """
        Initializes all annotations from dictionary.
        """
        self.__annotations = Annotations(*self.__image_dims, self.__annotation_type)
        for raw_annotation in self.__raw_annotations:
            self.__annotations.add(**raw_annotation)

    def _retrieve_meta_inf(self) -> dict:
        pass


class PackageHolder:

    BBOX_PARAMETERS = {"XMIN", "YMIN", "XMAX", "YMAX", "WIDTH", "HEIGHT", "XCENTER", "YCENTER"}
    KEYPOINT_PARAMETERS = {"KEYPOINT"}
    POLYGON_PARAMETERS = {"POLYGON"}
    MANDATORY_PARAMETERS = {"IMAGEREF"}
    OPTIONAL_PARAMETERS = {"LABELNAME", "LABELID", "CUSTOM"}

    def __init__(self):
        """
        Holds the unchanged annotation data as read-only data packages.
        """
        self.packages: List[DataPackage] = []
        self.boundary_type = None

    def __getitem__(self, item):
        """
        Implement functionality so data can be fetched by its meta-information. This functionality is mandatory for
        working filters.
        Implement extraction modes e.g. 'single-use', 'multi-use'
        """
        pass

    def load(
            self,
            image_folder_path: str,
            annotation_path: str,
            query: str,
            annotation_type: str,
            annotation_file_type: str,
            image_file_type: str
    ):
        """
        Loads raw annotations with AnnotationLoader. Preprocessed annotations to fit into DataPackage.
        Partially loads images for image size.
        """
        annotation_loader = AnnotationLoader(
            image_folder_path,
            annotation_path,
            query,
            annotation_type,
            annotation_file_type,
            image_file_type
        )
        raw_annotations = annotation_loader.load()
        preprocessed_raw_annots = self._preprocess_raw_annots(raw_annotations)
        for image_ref, raw_annotations in preprocessed_raw_annots.items():
            # TODO:
            # extracts image dimensions from image path
            image_path = f"{image_folder_path}/{image_ref}.{image_file_type}"
            image_dims = self.get_image_dims(image_path)
            self.packages.append(DataPackage(image_path, image_dims, raw_annotations, annotation_type))

    def _preprocess_raw_annots(self, raw_annotations: List[Dict[str, str]]):
        raw_annot_sample = raw_annotations[0]
        self.boundary_type = self._identify_boundary_type(raw_annot_sample)
        refactored_raw_annots = self._refactor_raw_annots(raw_annotations)
        return self.group_raw_annots(refactored_raw_annots)

    def _refactor_raw_annots(self, raw_annots):
        refactored_raw_annots: List[dict] = []
        for raw_annot in raw_annots:
            refactored_annot = {}
            match self.boundary_type:
                case "bbox":
                    refactored_annot["boundary_points"] = self._extract_bbox(raw_annot)
                case "keypoint":
                    refactored_annot["boundary_points"] = self._extract_keypoint(raw_annot)
                case "polygon":
                    refactored_annot["boundary_points"] = self._extract_polys(raw_annot)
            if "LABELID" in raw_annot.keys():
                refactored_annot["label_id"] = raw_annot["LABELID"]
            if "LABELNAME" in raw_annot.keys():
                refactored_annot["label_name"] = raw_annot["LABELNAME"]
            if "IMAGEREF" in raw_annot.keys():
                refactored_annot["image_ref"] = raw_annot["IMAGEREF"]
            else:
                raise ValueError("Unable to find image reference for annotations. Please adapt query and restart.")
            refactored_raw_annots.append(refactored_annot)
        return refactored_raw_annots

    def group_raw_annots(self, raw_annots: List[dict]) -> Dict[str, List[dict]]:
        """
        Groups raw annotations by their image reference.
        """
        sorted_raw_annots = sorted(raw_annots, key=itemgetter("image_ref"))
        grouped_raw_annots = {}
        current_image_ref = sorted_raw_annots[0]["image_ref"]
        current_group = []
        for raw_annot in sorted_raw_annots:
            if raw_annot["image_ref"] != current_image_ref:
                grouped_raw_annots[current_image_ref] = current_group
                current_image_ref = raw_annot["image_ref"]
                current_group = [raw_annot]
                continue
            del raw_annot["image_ref"]
            current_group.append(raw_annot)
        return grouped_raw_annots

    @staticmethod
    def _identify_boundary_type(raw_annot_sample) -> str:
        identified_boundary_types: List[str] = []
        parameters = list(raw_annot_sample.keys())
        if len(list(set(parameters).intersection(PackageHolder.BBOX_PARAMETERS))) > 0:
            identified_boundary_types.append("bbox")
        elif len(list(set(parameters).intersection(PackageHolder.KEYPOINT_PARAMETERS))) > 0:
            identified_boundary_types.append("keypoint")
        elif len(list(set(parameters).intersection(PackageHolder.POLYGON_PARAMETERS))) > 0:
            identified_boundary_types.append("polygon")
        assert len(identified_boundary_types) == 1, (f"Found ambiguous boundary types '{identified_boundary_types}'. "
                                                     f"Only one boundary type is allowed for annotations. "
                                                     f"Please adjust query and try again.")
        return identified_boundary_types[0]

    def _extract_bbox(self):
        pass

    def _extract_polys(self):
        pass

    def _extract_keypoint(self):
        pass

