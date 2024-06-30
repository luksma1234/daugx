from copy import deepcopy
from operator import itemgetter
from typing import Tuple, List, Dict

from daugx.core.augmentation.annotations import Annotations, Label
from daugx.utils.misc import read_img, shallow_load_img
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
        # TODO: Implement Meta Data
        pass


class PackageLoader:

    def __init__(self):
        pass

    # TODO: Make data hand-overs cleaner. Avoid loading data multiple times (like keys from the annot_dict).

    def load(
            self,
            image_folder_path: str,
            annotation_path: str,
            query: str,
            annotation_mode: str,
            annotation_file_type: str,
            image_file_type: str
    ) -> List[DataPackage]:
        """
        Loads raw annotations with AnnotationLoader. Preprocessed annotations to fit into DataPackage.
        Partially loads images for image size.
        Args:
            image_folder_path (str): Path to image folder
            annotation_path (str): Path to annotation file (if mode is onefile)
                                   or path to annotation directory (if mode is folder)
            query (str): Loading query for annotations
            annotation_mode (str): Load mode for annotations. Can be 'onefile' or 'folder'.
            annotation_file_type (str): File extension of annotation file(s)
            image_file_type (str): File extension of images
        Returns:
            (List[DataPackage]): List of all loaded data packages.
        """
        package_list = []
        annotation_loader = AnnotationLoader(
            image_folder_path,
            annotation_path,
            query,
            annotation_mode,
            annotation_file_type,
            image_file_type
        )
        raw_annotations = annotation_loader.load()
        preprocessed_raw_annots = self._preprocess_raw_annots(raw_annotations)
        for image_ref, raw_annotations in preprocessed_raw_annots.items():
            image_path = f"{image_folder_path}/{image_ref}.{image_file_type}"
            image_dims = shallow_load_img(image_path)
            package_list.append(DataPackage(image_path, image_dims, raw_annotations, annotation_mode))
        return package_list

    def _preprocess_raw_annots(self, raw_annotations: List[Dict[str, str]]):
        raw_annot_keys = list(raw_annotations[0].keys())
        self.boundary_type = self._identify_boundary_type(raw_annot_keys)
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

    @staticmethod
    def _identify_boundary_type(raw_annot_keys: list) -> str:
        """
        Identifies boundary type by mathing present keys with the parameters for bbox, polygon and keypoints.
        If any intersection is found, and the intersection is the only intersection, the boundary type is identified.
        Args:
            raw_annot_keys (dict): Keys from raw annotations
        """
        identified_boundary_types: List[str] = []
        if len(list(set(raw_annot_keys).intersection(PackageHolder.BBOX_PARAMETERS))) > 0:
            identified_boundary_types.append("bbox")
        elif len(list(set(raw_annot_keys).intersection(PackageHolder.KEYPOINT_PARAMETERS))) > 0:
            identified_boundary_types.append("keypoint")
        elif len(list(set(raw_annot_keys).intersection(PackageHolder.POLYGON_PARAMETERS))) > 0:
            identified_boundary_types.append("polygon")
        assert len(identified_boundary_types) == 1, (f"Found ambiguous boundary types '{identified_boundary_types}'. "
                                                     f"Only one boundary type is allowed for annotations. "
                                                     f"Please adjust query and try again.")
        return identified_boundary_types[0]

    @staticmethod
    def group_raw_annots(raw_annots: List[dict]) -> Dict[str, List[dict]]:
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

    def _extract_bbox(self, raw_annot: dict) -> np.ndarray:
        """
        Extracts all bounding box information from a raw annotation. Turns bbox into xyxy format.
        Args:
            raw_annot (dict): Raw annotation from loader
        """
        # 1. Check if necessary information to build bbox is present

        # 2. Load bbox

        # 3. Format bbox

    def _extract_polys(self):
        pass

    def _extract_keypoint(self):
        pass

    @staticmethod
    def _extract_bbox_keys(raw_annot: dict) -> List[str]:
        """
        Extracts all bounding box keys from a raw annot dictionary.
        Args:
            raw_annot (dict): Raw annotation from Loader
        Returns:
            (List[str]): List of all bbox keys
        """
        return list(set(raw_annot.keys()).intersection(PackageHolder.BBOX_PARAMETERS))

    def _is_raw_bbox(self, raw_annot: dict) -> bool:
        """
        Check if necessary information to build bbox is present
        BBOX_PARAMETERS = {"XMIN", "YMIN", "XMAX", "YMAX", "WIDTH", "HEIGHT", "XCENTER", "YCENTER"}
        """
        # TODO: Is this check even needed?
        n_pairs = 0
        raw_annot_keys = list(raw_annot.keys())
        if "XMIN" in raw_annot_keys and "XMAX" in raw_annot_keys:
            n_pairs += 1


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
        self.loader = PackageLoader()
        self.packages: List[DataPackage] = []
        self.boundary_type = None

    def __getitem__(self, item):
        """
        Implement functionality so data can be fetched by its meta-information. This functionality is mandatory for
        working filters.
        Implement extraction modes e.g. 'single-use', 'multi-use'
        """
        pass

    def load(self):
        """
        Passes values to loader.load(...) ... maybe something can be changed here to make value passing a bit cleaner
        """



































