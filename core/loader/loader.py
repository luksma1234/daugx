import os
import json
import csv

from typing import Optional, Tuple, List, Union, Dict
from pathlib import Path

from daugx.utils.misc import string_to_list, is_header

import xmltodict
import yaml


class Query:
    """
    A Query defines how exactly the data should be loaded. A query will be passed to the Loader.
    """

    ACCEPTED_KEYWORDS = ["XMIN", "XMAX", "YMIN", "YMAX", "WIDTH", "HEIGHT", "XCENTER", "YCENTER", "POLYGON", "KEYPOINT",
                         "LABELNAME", "LABELID", "IMAGEID"]

    def __init__(self, mode: str, query_string: str):
        """
        Args:
            mode (str): one of 'folder' or 'onefile' - specifies the base structure of the data
            query_string (str): the query string in the format '<Keyword> <Loading Query> ...'
                                Allowed Input Parameters are:
                                XMIN, XMAX, YMIN, YMAX, WIDTH, HEIGHT, XCENTER, YCENTER, POLYGON, KEYPOINT,
                                LABELNAME, LABELID, IMAGEREF

                                The Loading Query is made up of...
                                ... [] or {} to specify when to load from a list or a dictionary,
                                ... [nx] to indicate the iteration of a list without a specified index - where x is a
                                         number counting up from 0 and indicating which indices are counted equally
                                ... [1] to load the list entry at index 1
                                ... /filename/ reads the name of the file

                                As an example, the query string for the COCO Dataset would be:
                                'LABELID {annotations}[n][category_id] IMAGEID {annotations}[n]{image_id}
                                XMIN {annotations}[n]{bbox}[0] YMIN {annotations}[n]{bbox}[1] WIDTH
                                {annotations}[n]{bbox}[2] HEIGHT {annotations}[n]{bbox}[3]'
        """
        # TODO: Add CUSTOM as Input Parameter. Custom can handle multiple loading_strings separated by comma.
        #  An example would be: ... CUSTOM {annotations}[n]{is_grouped},{annotations}[n]{is_valid}



        self.__mode = mode
        self.__query_string = query_string
        self.__keywords: List[str] = []
        self.__loading_queries: List[str] = []
        self.__indexes: List[int] = []

        self._fail_counter = 0

        self._separate()
        self._validate()
        self._get_indexes()

    @property
    def mode(self):
        return self.__mode

    @property
    def keywords(self) -> List[str]:
        return self.__keywords

    @property
    def loading_queries(self) -> List[str]:
        return self.__loading_queries

    @property
    def indexes(self):
        return self._index_loading_queries()

    def _get_indexes(self):
        self.__indexes = [0] * max([loading_query.count("[n]") for loading_query in self.loading_queries])

    def _separate(self):
        """
        Separates query string into keywords and loading queries
        """
        query_list = string_to_list(self.__query_string)
        for index, item in query_list:
            if index % 2 == 0:
                self.__keywords.append(item)
            else:
                self.__loading_queries.append(item)

    def _validate(self):
        assert len(self.keywords) == len(self.loading_queries), ("Found more keywords that loading queries in Query. "
                                                                 "Please verify and try again.")
        for keyword in self.keywords:
            assert keyword in Query.ACCEPTED_KEYWORDS, f"Keyword '{keyword}' is unknown."

        # Check loading query for validity here

    def up_indexes(self, failed: bool):
        """
        Indexes of queries are initialized with 0. The indexes are iterated from the last index in the list upwards.
        An index is counted upwards as long as no IndexError with the current indexes occurs. An occurrence of an
        IndexError is handed over to this method with the 'failed' flag.
        If failed is set to true, the fail counter of this class is increased by one.The index of indexes at index
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
        partial_query = loading_query
        insert_index = 0
        n = 0
        while partial_query.count("[n]") > 0:
            find_index = partial_query.find("[n]")
            insert_index += find_index + 1
            partial_query = partial_query[find_index + 3:]
            loading_query = loading_query[:insert_index] + str(self.__indexes[n]) + loading_query[insert_index + 1:]
            insert_index += 2
            n += 1
        return loading_query

    def reset_indexes(self):
        """
        Resets current indexes to 0.
        """
        self._get_indexes()


class Loader:
    """
    The Loader loads the data from the disk using an input query.

    How is typical annotation data structured?
    There are two main differences:
    1. Annotations can come in the form of a folder with files, where each file includes annotations
       for one image.
    2. Annotations can also be one single file, where all annotations for all images are stored.

    On a deeper level, annotations are typically nested inside a dictionary and/or lists. The lists have
    an unknown amount of entries.

    Therefore, an annotation dataloader must have the following functionalities:
    1. Iterate over a list of unknown size - without providing any index
    2. Get an item from a list by a specified index
    3. Get an item from a dictionary by a specified key
    4. Differentiate between annotations in folders and one-file annotations
    5. Read the filename of a file
    5. Load data using these functionalities from all common annotation datatypes. (json, yaml, xml, txt, csv)

    How should data be loaded?
    - For each part of an annotation (category/label_name, label_id, x_min, x_max...)
      one query has to be created.
    - The user must specify the loading_mode [one-file, folder] - if the annotations are spread between files or are one-file
    - The user also has to provide the path to the folder/one-file
    - If loading_mode is folder - User must specify, what's the actual file type.

    """
    def __init__(
            self,
            image_folder_path: str,
            annotation_path: str,
            query: Query,
            annotation_file_type: str,
            image_file_type: str
    ):
        """
        Args:
            image_folder_path (str): Path to images folder
            annotation_path (str): Path to annotations file or folder
            query (Query): Loading Query
            annotation_file_type (str): File type for annotations. Accepted File types are: json, yaml, xml, txt, csv
            image_file_type (str): File type for images. Accepted file types are: jpg, png
        """
        self.image_folder_path = image_folder_path
        self.annotation_path = annotation_path
        self.query = query
        self.annotation_file_type = annotation_file_type
        self.image_file_type = image_file_type

    def load(self) -> List[Dict[str, str]]:
        """
        High level load method. Loads all annotation data from annotations file path.
        Returns:
             (List[Dict[str, str]]): List of all data retrieved by query
        """
        if self.query.mode == "folder":
            data = []
            for path in os.listdir(self.annotation_path):
                if path.endswith(self.annotation_file_type):
                    data.extend(self._load_from_query(path))
                    self.query.reset_indexes()
            return data
        elif self.query.mode == "onefile":
            return self._load_from_query(self.annotation_path)

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

    def _load_from_query(self, file_path: str):
        """
        Loads data on given file path from query.
        Args:
            file_path (str): Path to file to load
        """
        file, file_name = self._load_file(file_path)
        indexes = self.query.indexes
        load_list = []
        while indexes is not None:
            item_dict = {}
            try:
                for keyword, loading_query in zip(self.query.keywords, self.query.loading_queries):
                    if loading_query == "/filename/":
                        item_dict[keyword] = file_name
                        continue
                    feature = file
                    feature_query = loading_query
                    while feature_query:
                        if feature_query.startswith("["):
                            parameter_end = feature_query.find("]")
                            feature = feature[int(feature_query[1:parameter_end])]
                        else:
                            parameter_end = feature_query.find("}")
                            feature = feature[str(feature_query[1:parameter_end])]

                self.query.up_indexes(failed=False)
            except IndexError:
                # index error occurs when index is out of bounds
                self.query.up_indexes(failed=True)
            load_list.append(item_dict)
        return load_list

    def _load_file(self, file_path: str) -> Tuple[Union[list, dict], str]:
        match self.annotation_file_type:
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
