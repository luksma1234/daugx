import os
import json
import csv
import re

from typing import Tuple, List, Union, Dict
from pathlib import Path
from operator import itemgetter

from daugx.utils.misc import string_to_list, is_header
from daugx.core.data.data import DataPackage
import daugx.core.constants as c
from daugx.utils.misc import img_dims, list_intersection
from daugx.core.augmentation.annotations import Annotations

import xmltodict
import yaml
import numpy as np


class Query:
    """
    A Query defines how exactly the data should be loaded. A query will be passed to the Loader.
    """

    def __init__(self, mode: str, query_string: str):
        """
        Args:
            mode (str): one of 'directory' or 'onefile' - specifies the base structure of the data
            query_string (str): the query string in the format '<Keyword> <Loading Query> ...'
                                Allowed Input Parameters are:
                                XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, XCENTER, YCENTER, POLYGON, KEYPOINT,
                                LABELNAME, LABELID, IMAGEREF, CUSTOM

                                The Parameter 'CUSTOM' makes an exception. Here multiple loading strings can be defined,
                                which are then separated by a comma. An example would be:
                                ... CUSTOM {annotations}[n]{is_grouped},{annotations}[n]{is_valid} ...

                                The Loading Query is made up of...
                                ... [] or {} to specify when to load from a list or a dictionary,
                                ... [n] to indicate the iteration of a list without a specified index
                                ... [1] to load the list entry at index 1
                                ... /filename/ reads the name of the file

                                As an example, the query string for the COCO Dataset would be:
                                'LABELID {annotations}[n][category_id] IMAGEID {annotations}[n]{image_id}
                                XMIN {annotations}[n]{bbox}[0] YMIN {annotations}[n]{bbox}[1] WIDTH
                                {annotations}[n]{bbox}[2] HEIGHT {annotations}[n]{bbox}[3]'
        """
        self.__mode = mode
        self.__query_string = query_string
        self.__keywords: List[str] = []
        self.__loading_queries: List[str] = []
        self.__indexes: List[int] = []

        self._fail_counter = 0

        self._separate()
        self._validate()
        self._get_indexes()

        assert len(self.keywords) == len(self.loading_queries), ("Found more keywords that loading queries in Query. "
                                                                 "Please verify and try again.")

    @property
    def mode(self):
        return self.__mode

    @property
    def keywords(self) -> List[str]:
        return self.__keywords

    @property
    def loading_queries(self) -> List[str]:
        return self._index_loading_queries()

    @property
    def indexes(self):
        return self.__indexes

    def _get_indexes(self):
        self.__indexes = [0] * max(
            [loading_query.count(c.QUERY_UNDEFINED_ITERATOR) for loading_query in self.__loading_queries]
        )

    def _separate(self):
        """
        Separates query string into keywords and loading queries.
        """
        query_list = string_to_list(self.__query_string)
        for index, item in enumerate(query_list):
            if index % 2 == 0:
                self.__keywords.append(item)
            else:
                self.__loading_queries.append(item)

    def _validate(self):
        """
        Validates all found keywords of query string.
        Raises:
            AssertionError: If keyword is not part of the QUERY_KEYWORDS_LIST.
        """
        for keyword in self.keywords:
            assert keyword in c.QUERY_KEYWORDS_LIST, f"Keyword '{keyword}' is unknown."

        # Check loading query for validity here

    def _handle_custom(self):
        """
        Handles all custom queries. Separates custom queries and lists them with their query index.
        e.g.: CUSTOM {example}[n][1],{example}[n][2]
        -> {
            ...,
            CUSTOM_0: "{example}[n][1]",
            CUSTOM_1: "{example}[n][2]"
           }
        """
        if c.QUERY_CUSTOM in self.keywords:
            custom_index = self.keywords.index(c.QUERY_CUSTOM)
            custom_query = self.__loading_queries[custom_index]
            del self.keywords[custom_index]
            del self.__loading_queries[custom_index]
            # If multiple queries are listed as custom, the queries must be separated by a ','
            # Keep in mind that spaces are used to separate query keywords and query-blocks.
            # Therefore, do not use ', ' as separator.
            custom_queries = custom_query.split(",")
            for index, query in enumerate(custom_queries):
                self.keywords.append(f"{c.QUERY_CUSTOM}_{index}")
                self.__loading_queries.append(query)

    def up_indexes(self, failed: bool):
        """
        Indexes of queries are initialized as 0. The indexes are iterated from the last index in the list upwards.
        An index is counted upwards as long as no IndexError with the current indexes occurs. An occurrence of an
        IndexError is handed over to this method using the 'failed' flag.
        If failed is set to true, the fail counter of this class is increased by one. The index of indexes at index
        self.fail_counter + 1 is then increased by one. Sets indexes to None if fail_counter exceeds length of indexes.
        Args:
            failed (bool): Flag to indicate weather an Index error occurred with current indexes.
        """
        if not failed:
            self._fail_counter = 0
            self.__indexes[-1] += 1
        else:
            self._fail_counter += 1
            if self._fail_counter == len(self.indexes):
                self.__indexes = None
                return
            self.__indexes[-self._fail_counter:] = [0] * self._fail_counter
            self.__indexes[-self._fail_counter - 1] += 1

    def _index_loading_queries(self):
        """
        Injects indexes into all loading queries
        """
        indexed_loading_queries = []
        for loading_query in self.__loading_queries:
            indexed_loading_queries.append(self._index_loading_query(loading_query))
        return indexed_loading_queries

    def _index_loading_query(self, loading_query: str):
        """
        Injects indexes to one loading query
        """
        indexed_loading_query = ""
        prev_index = 0
        index_occurrences = [m.start() for m in re.finditer(c.REGEX_QUERY_UNDEFINED_ITERATOR, loading_query)]
        for list_index, index_occurrence in enumerate(index_occurrences):
            indexed_loading_query += (loading_query[prev_index:index_occurrence] + f"[{self.__indexes[list_index]}]")
            prev_index = index_occurrence + 3
        indexed_loading_query += loading_query[prev_index:]
        return indexed_loading_query

    def reset_indexes(self):
        """
        Resets current indexes to 0.
        """
        self._get_indexes()


class InitialLoader:
    """
    The Loader loads the data from the disk using an input query.

    How is typical annotation data structured?
    There are two main differences:
    1. Annotations can come in the form of a directory with files, where each file includes annotations
       for one image.
    2. Annotations can also be one single file, where all annotations for all images are stored.

    On a deeper level, annotations are typically nested inside a dictionary and/or lists. The lists have
    an unknown amount of entries.

    Therefore, an annotation dataloader must have the following functionalities:
    1. Iterate over a list of unknown size - without providing any index
    2. Get an item from a list by a specified index
    3. Get an item from a dictionary by a specified key
    4. Differentiate between annotations in directories and one-file annotations
    5. Read the filename of a file
    5. Load data using these functionalities from all common annotation datatypes. (json, yaml, xml, txt, csv)

    How should data be loaded?
    - For each part of an annotation (category/label_name, label_id, x_min, x_max...)
      one query has to be created.
    - The user must specify the loading_mode [one-file, directory] - if the annotations are spread between files or are one-file
    - The user also has to provide the path to the directory/one-file
    - If loading_mode is directory - User must specify, what's the actual file type.

    """
    def __init__(
            self,
            img_dir_path: str,
            annot_path: str,
            query: str,
            annot_mode: str,
            annot_file_type: str,
            img_file_type: str
    ):
        """
        Args:
            img_dir_path (str): Path to images directory
            annot_path (str): Path to annotations file or directory
            query (str): Loading Query
            annot_mode: Load mode for annotations. Can be 'onefile' or 'directory'.
            annot_file_type (str): File type for annotations. Accepted File types are: json, yaml, xml, txt, csv
            img_file_type (str): File type for images. Accepted file types are: jpg, png
        """
        self.img_dir_path = img_dir_path
        self.annot_path = annot_path
        self.annot_mode = annot_mode
        self.query = Query(query_string=query, mode=self.annot_mode)
        self.annot_file_type = annot_file_type
        self.img_file_type = img_file_type
        self.boundary_type = self._identify_boundary_type()
        self.__current_working_file = None
        if self.boundary_type == c.BOUNDARY_TYPE_BBOX:
            self.bbox_keywords = self._extract_bbox_keywords()
            self.bbox_strategy = self._get_bbox_strategy(self.bbox_keywords)

    def load(self) -> List[DataPackage]:
        """
        Loads all annotations defined in query.
        Returns:
            (List[DataPackage]): All Annotations of all image references.
        """
        raw_annots = self._load_raw_annots()
        return self._raw_to_packages(raw_annots)

    @property
    def current_working_file(self):
        return self.__current_working_file

    def _load_raw_annots(self) -> List[Dict[str, str]]:
        """
        High level load method for annotations. Loads all annotation data from annotations file path.
        Returns:
             (List[Dict[str, str]]): List of all data retrieved by query
             dict schema: {
                             <query_key>: <data (str)>,
                             ...
                          }
        """
        if self.query.mode == c.QUERY_MODE_DIRECTORY:
            data = []
            for path in os.listdir(self.annot_path):
                if path.endswith(self.annot_file_type):
                    data.extend(self._load_from_query(path))
                    self.query.reset_indexes()
        elif self.query.mode == c.QUERY_MODE_ONE_FILE:
            data = self._load_from_query(self.annot_path)
        else:
            raise NotImplementedError(f"Query mode '{self.query.mode}' is unknown.")
        return data

    @staticmethod
    def _load_xml(file_path: str) -> Tuple[dict, str]:
        """
        Loads a xml file. Returns Tuple of xml content as dict and file name as string.
        Args:
            file_path (str): Path to xml file
        """
        path = Path(file_path)
        with open(path, "r") as f:
            return xmltodict.parse(f.read()), path.stem

    @staticmethod
    def _load_json(file_path: str) -> Tuple[dict, str]:
        """
        Loads a json file. Returns Tuple of json content as dict and file name as string.
        Args:
            file_path (str): Path to json file
        """
        path = Path(file_path)
        with open(path, "r") as f:
            return json.load(f), path.stem

    @staticmethod
    def _load_yaml(file_path: str) -> Tuple[dict, str]:
        """
        Loads a yaml file. Returns Tuple of yaml content as dict and file name as string.
        Args:
            file_path (str): Path to yaml file
        """
        path = Path(file_path)
        with open(path, "r") as f:
            return yaml.safe_load(f), path.stem

    @staticmethod
    def _load_csv(file_path: str) -> Tuple[Union[List[Dict[str, str]], List[List[str]]], str]:
        """
        Loads a csv file. Returns Tuple of csv content as dict or list and file name as string.
        Args:
            file_path (str): Path to csv file
        """
        path = Path(file_path)
        has_header = False
        content_list = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for index, line in enumerate(reader):
                if index == 0:
                    if is_header(line):
                        has_header = True
                        header = line
                        continue
                if has_header:
                    content_dict = {}
                    for key, value in zip(header, line):
                        content_dict[key] = value
                    content_list.append(content_dict)
                else:
                    content_list.append(line)
        return content_list, path.stem

    @staticmethod
    def _load_txt(file_path: str) -> Tuple[Union[List[Dict[str, str]], List[List[str]]], str]:
        """
        Loads a txt file. Returns Tuple of txt content as dict or list and file name as string.
        Args:
            file_path (str): Path to txt file
        """
        path = Path(file_path)
        has_header = False
        content_list = []
        with open(path, "r") as f:
            reader = f.readlines()
            for index, str_line in enumerate(reader):
                line = string_to_list(str_line)
                if index == 0:
                    if is_header(line):
                        has_header = True
                        header = line
                if has_header:
                    content_dict = {}
                    for key, value in zip(header, line):
                        content_dict[key] = value
                    content_list.append(content_dict)
                else:
                    content_list.append(line)
        return content_list, path.stem

    def _load_from_query(self, file_path: str) -> List[Dict[str, str]]:
        """
        Loads data on given file path from query.
        Args:
            file_path (str): Path to file to load
        Returns:
            (List[Dict[str, str]): List of all annotations on given file path.
            dict schema: {
                             <query_key>: <data (str)>,
                             ...
                         }
        """
        self.__current_working_file, file_name = self._load_file(file_path)
        load_list = []
        while self.query.indexes is not None:
            item_dict = {}
            for keyword, loading_query in zip(self.query.keywords, self.query.loading_queries):
                if loading_query == c.QUERY_CURRENT_FILE_NAME:
                    item_dict[keyword] = file_name
                    continue
                if not loading_query.startswith("[") and not loading_query.startswith("{"):
                    # its a constant value
                    item_dict[keyword] = loading_query
                    continue
                feature = self._get_item_by_query(loading_query, self.current_working_file)
                if feature is None:
                    break
                item_dict[keyword] = feature
            else:
                # ups indexes with failed as false only if loop runs successfully
                self.query.up_indexes(failed=False)
                load_list.append(item_dict)
        return load_list

    def _get_item_by_query(self, feature_query: str, feature: Union[dict, list]):
        index_list = self._query_to_index_list(feature_query)
        for index in index_list:
            try:
                feature = feature[index]
            except (IndexError, KeyError, TypeError):
                self.query.up_indexes(failed=True)
                return None
        return feature

    @staticmethod
    def _query_to_index_list(loading_query: str):
        index_list = []
        while loading_query:
            if loading_query.startswith("["):
                parameter_end = loading_query.find("]")
                index_list.append(int(loading_query[1:parameter_end]))
            elif loading_query.startswith("{"):
                parameter_end = loading_query.find("}")

                index_list.append(str(loading_query[1:parameter_end]))
            else:
                raise ValueError("Missing or invalid brackets in loading query. "
                                 "Please verify your query and try again.")
            loading_query = loading_query[parameter_end + 1:]
        return index_list

    def _load_file(self, file_path: str) -> Tuple[Union[list, dict], str]:
        match self.annot_file_type:
            case "csv":
                return self._load_csv(file_path)
            case "json":
                return self._load_json(file_path)
            case "txt":
                return self._load_txt(file_path)
            case "xml":
                return self._load_xml(file_path)
            case "yaml":
                return self._load_yaml(file_path)

    def _raw_to_packages(self, raw_annots: List[Dict[str, str]]) -> List[DataPackage]:
        """
        Transforms raw loaded annotations into a list of data packages.
        Args:
            raw_annots (List[Dict[str, str]]): Raw loaded annotations
        Returns:
            (List[DataPackage])
        """
        package_list: List[DataPackage] = []
        annot_kwargs = self._refactor_raw_annots(raw_annots)
        grouped_annot_kwargs = self._group_by_img_ref(annot_kwargs)
        for image_ref, annot_kwargs in grouped_annot_kwargs.items():
            image_path = self._image_path_from_ref(image_ref)
            image_dims = img_dims(image_path)
            annotations = Annotations(*image_dims, self.boundary_type)
            package_list.append(DataPackage(image_path, annotations))
        return package_list

    def _image_path_from_ref(self, image_ref: str):
        if not image_ref.endswith(f".{self.img_file_type}"):
            return f"{self.img_dir_path}/{image_ref}.{self.img_file_type}"
        else:
            return f"{self.img_dir_path}/{image_ref}"

    def _grouped_annot_kwargs_to_annotations(self, annot_kwargs_list: list, image_dims: Tuple[int, int]) -> Annotations:
        """
        Initializes all annotations from a grouped annot kwarg list.
        Args:
            annot_kwargs_list (dict): List of annot kwargs. Contains all annotation kwargs of one image reference.
            image_dims (Tuple[int, int]): width and height of referenced image
        Returns:
            (Annotations): Annotations of image reference
        """
        # Don't know if that's working... maybe content of image_dims cant be unpacked like that
        annotations = Annotations(*image_dims, self.boundary_type)
        for single_annot_kwargs in annot_kwargs_list:
            annotations.add(**single_annot_kwargs)
        return annotations

    def _refactor_raw_annots(self, raw_annots: List[Dict[str, str]]) -> List[dict]:
        """
        Refactors list of raw annotation dictionaries to fit the kwargs of the Annotation class.
        Args:
            raw_annots (List[Dict[str, str]]): Raw loaded annotations
        Returns:
            (List[dict]): List of dictionaries, which can be fed as kwargs to the Annotation class
        Raises:
            ValueError: If image reference could not be found in the loading Query
            AssertionError: If none of annotation id or name was found in the loading Query
        """
        annot_kwargs: List[dict] = []
        for raw_annot in raw_annots:
            refactored_annot = {c.DICTIONARY_KEY_BOUNDARY_POINTS: self._extract_boundary(raw_annot)}
            if c.QUERY_LABEL_ID in raw_annot.keys():
                refactored_annot[c.DICTIONARY_KEY_LABEL_ID] = raw_annot[c.QUERY_LABEL_ID]
            if c.QUERY_LABEL_NAME in raw_annot.keys():
                refactored_annot[c.DICTIONARY_KEY_LABEL_NAME] = raw_annot[c.QUERY_LABEL_NAME]
            if c.QUERY_IMAGE_REF in raw_annot.keys():
                refactored_annot[c.DICTIONARY_KEY_IMAGE_REF] = raw_annot[c.QUERY_IMAGE_REF]
            else:
                raise ValueError("Unable to find image reference for annotations. Please adapt query and restart.")
            assert c.DICTIONARY_KEY_LABEL_ID in refactored_annot or c.DICTIONARY_KEY_LABEL_NAME in refactored_annot, \
                (f"Annotations must either have an id or a name. Please adapt your loading query and add "
                 f"{c.QUERY_LABEL_ID} or {c.QUERY_LABEL_NAME}.")
            annot_kwargs.append(refactored_annot)
        return annot_kwargs

    def _extract_boundary(self, raw_annot: Dict[str, str]) -> np.ndarray:
        """
        Extracts a boundary according to the boundary type from a raw annotation.
        Args:
            raw_annot ([Dict[str, str]]): Single raw annotation
        Returns:
            (np.ndarray): Boundary of raw annotation
        """
        match self.boundary_type:
            case c.BOUNDARY_TYPE_BBOX:
                return self._extract_bbox(raw_annot)
            case c.BOUNDARY_TYPE_KEYPOINT:
                return self._extract_keypoint(raw_annot)
            case c.BOUNDARY_TYPE_POLYGON:
                return self._extract_polys(raw_annot)

    def _identify_boundary_type(self) -> str:
        """
        Identifies the type of boundary inside the query. Does not check for ambiguities in boundary types.
        Matches the first boundary type identified in query.
        Returns:
            (str): Boundary type of annotations loaded by query
        Raises:
            ValueError: If no boundary type could be matched
        """
        if len(list_intersection(self.query.keywords, c.QUERY_BBOX_KEYWORDS_LIST)) > 0:
            return c.BOUNDARY_TYPE_BBOX
        if len(list_intersection(self.query.keywords, c.QUERY_KEYPOINT_KEYWORDS_LIST)) > 0:
            return c.BOUNDARY_TYPE_KEYPOINT
        if len(list_intersection(self.query.keywords, c.QUERY_POLYGON_KEYWORDS_LIST)) > 0:
            return c.BOUNDARY_TYPE_POLYGON
        raise ValueError("Unable to match any boundary type. Please verify your query and try again.")

    @staticmethod
    def _group_by_img_ref(raw_annots: List[dict]) -> Dict[str, List[dict]]:
        """
        Groups raw annotations by their image reference.
        Args:
            raw_annots (List[dict]): Raw annotations list
        Returns:
            (Dict[str, List[dict]]): Dict with image references as keys and lists of raw annotations as items
        """
        sorted_raw_annots = sorted(raw_annots, key=itemgetter(c.DICTIONARY_KEY_IMAGE_REF))
        grouped_raw_annots = {}
        current_image_ref = sorted_raw_annots[0][c.DICTIONARY_KEY_IMAGE_REF]
        current_group = []
        for raw_annot in sorted_raw_annots:
            if raw_annot[c.DICTIONARY_KEY_IMAGE_REF] != current_image_ref:
                grouped_raw_annots[current_image_ref] = current_group
                current_image_ref = raw_annot[c.DICTIONARY_KEY_IMAGE_REF]
                current_group = [raw_annot]
                continue
            del raw_annot[c.DICTIONARY_KEY_IMAGE_REF]
            current_group.append(raw_annot)
        return grouped_raw_annots

    def _extract_bbox(self, raw_annot: dict) -> np.ndarray:
        """
        Extracts all bounding box information from a raw annotation. Turns bbox into xyxy format.
        Args:
            raw_annot (dict): Raw annotation from loader
        Returns:
            (np.ndarray): Numpy array of bounding box in xyxy format
        """
        match self.bbox_strategy:

            case [c.KEYWORD_PAIR_XY_MIN, c.KEYWORD_PAIR_XY_MAX] | [c.KEYWORD_PAIR_XY_MAX, c.KEYWORD_PAIR_XY_MIN]:
                return np.array([
                    [raw_annot[c.QUERY_X_MIN], raw_annot[c.QUERY_Y_MIN]],
                    [raw_annot[c.QUERY_X_MAX], raw_annot[c.QUERY_Y_MAX]]
                ], np.float32)

            case [c.KEYWORD_PAIR_XY_MIN, c.KEYWORD_PAIR_WH] | [c.KEYWORD_PAIR_WH, c.KEYWORD_PAIR_XY_MIN]:
                return np.array([
                    [raw_annot[c.QUERY_X_MIN], raw_annot[c.QUERY_Y_MIN]],
                    [float(raw_annot[c.QUERY_X_MIN]) + float(raw_annot[c.QUERY_WIDTH]),
                     float(raw_annot[c.QUERY_Y_MIN]) + float(raw_annot[c.QUERY_HEIGHT])]
                ], np.float32)

            case [c.KEYWORD_PAIR_XY_MIN, c.KEYWORD_PAIR_CENTER] | [c.KEYWORD_PAIR_CENTER, c. KEYWORD_PAIR_XY_MIN]:
                return np.array([
                    [raw_annot[c.QUERY_X_MIN], raw_annot[c.QUERY_Y_MIN]],
                    [float(raw_annot[c.QUERY_X_MIN]) +
                     (float(raw_annot[c.QUERY_X_CENTER]) - float(raw_annot[c.QUERY_X_MIN])) * 2,
                     float(raw_annot[c.QUERY_Y_MIN]) +
                     (float(raw_annot[c.QUERY_Y_CENTER]) - float(raw_annot[c.QUERY_Y_MIN])) * 2]
                ], np.float32)

            case [c.KEYWORD_PAIR_WH, c.KEYWORD_PAIR_CENTER] | [c.KEYWORD_PAIR_CENTER, c.KEYWORD_PAIR_WH]:
                return np.array([
                    [float(raw_annot[c.QUERY_X_CENTER]) - (float(raw_annot[c.QUERY_WIDTH]) / 2),
                     float(raw_annot[c.QUERY_Y_CENTER]) - (float(raw_annot[c.QUERY_HEIGHT]) / 2)],
                    [float(raw_annot[c.QUERY_X_CENTER]) + (float(raw_annot[c.QUERY_WIDTH]) / 2),
                     float(raw_annot[c.QUERY_Y_CENTER]) + (float(raw_annot[c.QUERY_HEIGHT]) / 2)]
                ], np.float32)

            case [c.KEYWORD_PAIR_WH, c.KEYWORD_PAIR_XY_MAX] | [c.KEYWORD_PAIR_XY_MAX, c.KEYWORD_PAIR_WH]:
                return np.array([
                    [float(raw_annot[c.QUERY_X_MAX]) - float(raw_annot[c.QUERY_WIDTH]),
                     float(raw_annot[c.QUERY_Y_MAX]) - float(raw_annot[c.QUERY_HEIGHT])],
                    [raw_annot[c.QUERY_X_MAX], raw_annot[c.QUERY_Y_MAX]]
                ], np.float32)

            case [c.KEYWORD_PAIR_XY_MAX, c.KEYWORD_PAIR_CENTER] | [c.KEYWORD_PAIR_CENTER, c. KEYWORD_PAIR_XY_MAX]:
                return np.array([
                    [float(raw_annot[c.QUERY_X_MAX]) -
                     (float(raw_annot[c.QUERY_X_MAX]) - float(raw_annot[c.QUERY_X_CENTER])) * 2,
                     float(raw_annot[c.QUERY_Y_MAX]) -
                     (float(raw_annot[c.QUERY_Y_MAX]) - float(raw_annot[c.QUERY_Y_CENTER])) * 2],
                    [raw_annot[c.QUERY_X_MAX], raw_annot[c.QUERY_Y_MAX]]
                ], np.float32)

    def _extract_bbox_keywords(self) -> List[str]:
        """
        Extracts all bounding box keys from the query
        Returns:
            (List[str]): List of all bbox keys
        """
        return list_intersection(self.query.keywords, c.QUERY_BBOX_KEYWORDS_LIST)

    @staticmethod
    def _get_bbox_strategy(bbox_keywords: List[str]) -> List[str]:
        """
        Gets bbox strategy by checking keywords. Strategy is later used to transform any type of bbox into a xyxy bbox.
        Returns:
            (List[str]): List of found keyword pairs. Results in a strategy.
        """
        bbox_strategy = []
        if c.QUERY_X_MIN in bbox_keywords and c.QUERY_Y_MIN in bbox_keywords:
            bbox_strategy.append(c.KEYWORD_PAIR_XY_MIN)
        if c.QUERY_WIDTH in bbox_keywords and c.QUERY_HEIGHT in bbox_keywords:
            bbox_strategy.append(c.KEYWORD_PAIR_WH)
        if c.QUERY_X_CENTER in bbox_keywords and c.QUERY_Y_CENTER in bbox_keywords:
            bbox_strategy.append(c.KEYWORD_PAIR_CENTER)
        if c.QUERY_X_MAX in bbox_keywords and c.QUERY_Y_MAX in bbox_keywords:
            bbox_strategy.append(c.KEYWORD_PAIR_XY_MAX)
        assert len(bbox_strategy) >= 2, \
            (f"Insufficient bbox keyword pairs found in query. Needs two pairs, found {len(bbox_strategy)}. Adapt "
             f"loading query by adding or complete any unused or incomplete keyword pair with its loading instructions "
             f"of '{c.QUERY_X_MIN},{c.QUERY_Y_MIN}', '{c.QUERY_X_MAX}, {c.QUERY_Y_MAX}', '{c.QUERY_X_CENTER}, "
             f"{c.QUERY_Y_CENTER}' or '{c.QUERY_WIDTH}, {c.QUERY_HEIGHT}'")
        return bbox_strategy[:2]

    @staticmethod
    def _extract_polys(raw_annot: dict) -> np.ndarray:
        """
        Extracts all polygon information from a raw annotation. Assumes polys to be in x,y,x,y format inside a list.
        Args:
            raw_annot (dict): Raw annotation from loader
        Returns:
            (np.ndarray): Numpy array of polygon
        """
        poly_string: str = str(raw_annot[c.QUERY_POLYGON])
        # remove spaces
        poly_string = poly_string.replace(" ", "")
        # remove leading and trailing brackets
        poly_string = poly_string.strip("[]")
        polys = poly_string.split(",")
        return np.array(polys, np.float32).reshape((int(len(polys) / 2), 2))

    @staticmethod
    def _extract_keypoint(raw_annot: dict) -> np.ndarray:
        """
        Extracts a keypoint from a raw annotation.
        Args:
            raw_annot (dict): Raw annotation from loader
        Returns:
            (np.ndarray): Numpy array of keypoint
        """
        keypoint_string: str = raw_annot[c.QUERY_KEYPOINT]
        keypoint_string = keypoint_string.strip("[]")
        keypoint_string = keypoint_string.replace(" ", "")
        return np.array(keypoint_string.split(","))
