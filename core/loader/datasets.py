import json
import csv
import os
from tqdm import tqdm
from operator import itemgetter
import xmltodict
import yaml
import pathlib
# This is important, because some annotations are given as percentage values and have to be recalculated
import imagesize


class AnnotationAdaptor:
    """
    Adaptor for loading different annotation file types into the same format.
    Adaptor takes a config file.
    For configuration of the target location, the following keywords are available:
    (square brackets define user input of type given inside the brackets
     e.g.: .[Integer] -> .1)

    .n              -> implicit iterator
                    | iterates over list, returns value for each iteration
    .name           -> file name of current file
                    | returns file name without extension (only works for iterable files)
    .[Integer]key   -> key at given position - e.g.: .3key
                    | returns the key at specified position
    .[Integer]      -> explicit iterator - e.g.: .1
                    | returns value for specified iteration (recommended)
                      can also be used to get value for key at specified position (not recommended).
    [string]        -> key - e.g.: bbox
                    | returns value for specified key
    """
    def __init__(self, configs: dict):
        self.basic_configs = configs["basic"]
        self.advanced_configs = configs["advanced"]
        self.annots = None
        self.annots_file_names = None
        self.accepted_dtypes = ["json", "csv", "xml", "txt", "yaml"]

    def load_annots(self):
        self.__list_configs()
        self.__validate_configs()
        self.annots, self.annots_file_names = self.__load_annot_files()
        n_iters = self.get_n_iters()
        current_index = [0] * n_iters
        main_index = self.__get_main_index()
        annots = []
        with tqdm(total=main_index) as pbar:
            while current_index[0] < main_index:
                if n_iters == 1:
                    index_item = self.__get_items(current_index)
                    annots.append(index_item)
                else:
                    while True:
                        index_item = self.__get_items(current_index)
                        if index_item is None:
                            # up index updater to higher level (lower value)
                            index_updater -= 1
                            if index_updater == 0:
                                # break if index_updater reaches main_index
                                break
                        else:
                            annots.append(index_item)
                            # reset index updater
                            index_updater = len(current_index) - 1
                        current_index = self.__update_index_list(current_index, index_updater)
                # update main index
                current_index = self.__update_index_list(current_index, 0)
                pbar.update(1)
        self.annots = annots
        self.annots = self.__sort_annots()
        self.annots = self.__group_annots()
        return self.annots

    def __get_items(self, index_list: list):
        items = {}
        for key in self.advanced_configs.keys():
            list_index = 0
            if not self.advanced_configs[key]:
                continue
            item = self.annots.copy()
            for item_index in self.advanced_configs[key]:
                item, list_index, continuation = self.__get_index_item(
                    item=item, index=item_index, index_list=index_list, current_list_index=list_index
                )
                if continuation == "continue":
                    continue
                if continuation == "break":
                    break
                if continuation == "return":
                    return None
            items[key] = item
        if not items:
            raise ValueError("Found empty item.")
        return items

    def __get_main_index(self):
        main_index = 0
        for key in self.advanced_configs:
            if not self.advanced_configs[key]:
                continue
            if ".n" not in self.advanced_configs[key]:
                continue
            items = self.annots.copy()
            for index, item_index in enumerate(self.advanced_configs[key]):
                if item_index == ".n":
                    if main_index == 0:
                        main_index = len(items)
                        break
                    else:
                        assert main_index == len(items), f"Main Index has to match for every item. " \
                                                        f"Found indices {main_index} and {len(items)}."
                        break
                items, _, continuation = self.__get_index_item(
                    item=items, index=item_index, index_list=[0], current_list_index=0
                )
                if continuation == "continue":
                    continue
                else:
                    raise ValueError(f"Error in calculating max_index for '{key}': '{self.advanced_configs[key]}'.")
        return main_index

    def get_n_iters(self):
        n_iters = 0
        for key in self.advanced_configs:
            if not self.advanced_configs[key]:
                continue
            config_iters = 0
            for index, item_index in enumerate(self.advanced_configs[key]):
                if item_index == ".n":
                    config_iters += 1
            if config_iters > n_iters:
                n_iters = config_iters
        if n_iters == 0:
            raise ValueError("Found no iterable in adaptor config.")
        return n_iters

    @staticmethod
    def __update_index_list(index_list: list, update_index: int):
        index_list[update_index] += 1
        for index in range(update_index + 1, len(index_list)):
            index_list[index] = 0
        return index_list

    @staticmethod
    def __list_equality(a: list, b: list):
        for a_i, b_i in zip(a, b):
            if a_i == b_i:
                continue
            return False
        else:
            return True

    def __load_annot_files(self):
        annot_path = self.basic_configs["annot_path"]
        file_type = self.basic_configs["file_type"]
        is_dir = self.basic_configs["is_dir"]
        if file_type not in self.accepted_dtypes:
            raise TypeError(f"Unaccepted file type for annotations: {file_type}. "
                            f"Accepted file types are: {self.accepted_dtypes}")
        if file_type == "json":
            return self.__load_json_annots(annot_path, is_dir)
        if file_type == "csv":
            return self.__load_csv_annots(annot_path, is_dir)
        if file_type == "xml":
            return self.__load_xml_annots(annot_path, is_dir)
        if file_type == "txt":
            return self.__load_txt_annots(annot_path, is_dir)
        if file_type == "yaml":
            return self.__load_yaml_annots(annot_path, is_dir)

    @staticmethod
    def __load_xml_annots(annot_path: str, is_dir: bool):
        if not is_dir:
            with open(annot_path, "r") as f:
                annot = xmltodict.parse(f.read())
            return annot, None
        else:
            annots = []
            annots_file_names = []
            for file in os.listdir(annot_path):
                if file.endswith(".xml"):
                    with open(f"{annot_path}/{file}", "r") as f:
                        annot = xmltodict.parse(f.read())
                    annots.append(annot)
                    annots_file_names.append(file.removesuffix(".xml"))
                else:
                    continue
            return annots, annots_file_names

    @staticmethod
    def __load_json_annots(annot_path: str, is_dir: bool):
        if not is_dir:
            with open(annot_path, "r") as f:
                annot = json.load(f)
            return annot, None
        else:
            annots = []
            annots_file_names = []
            for file in os.listdir(annot_path):
                if file.endswith(".json"):
                    with open(f"{annot_path}/{file}", "r") as f:
                        annot_file = json.load(f)
                    annots.append(annot_file)
                    annots_file_names.append(file.removesuffix(".json"))
                else:
                    continue
            return annots, annots_file_names

    @staticmethod
    def __load_yaml_annots(annot_path: str, is_dir: bool):
        if not is_dir:
            with open(annot_path, "r") as f:
                annot = yaml.safe_load(f)
            return annot, None
        else:
            annots = []
            annots_file_names = []
            for file in os.listdir(annot_path):
                if file.endswith(".json"):
                    with open(f"{annot_path}/{file}", "r") as f:
                        annot_file = yaml.safe_load(f)
                    annots.append(annot_file)
                    annots_file_names.append(file.removesuffix(".json"))
                else:
                    continue
            return annots, annots_file_names

    def __load_csv_annots(self, annot_path: str, is_dir: bool):
        annots = []
        has_header = False
        header = []
        annots_file_names = []
        if not is_dir:
            with open(annot_path, "r") as f:
                annot_reader = csv.reader(f)
            for index, line in enumerate(annot_reader):
                if index == 0:
                    if self.__is_header(line):
                        has_header = True
                        header = line
                if has_header:
                    annot_dict = {}
                    for key, value in zip(header, line):
                        annot_dict[key] = value
                    annots.append(annot_dict)
                else:
                    annots.append(line)
            return annots, None
        else:
            for file in os.listdir(annot_path):
                if file.endswith(".csv"):
                    with open(f"{annot_path}/{file}", "r") as f:
                        annot_reader = csv.reader(f)

                    for index, line in enumerate(annot_reader):
                        if index == 0:
                            if self.__is_header(line):
                                has_header = True
                                header = line
                                continue
                        if has_header:
                            annot_dict = {}
                            for key, value in zip(header, line):
                                annot_dict[key] = value
                            annots.append(annot_dict)
                        else:
                            annots.append(line)
                            annots_file_names.append(file.removesuffix(".csv"))
                else:
                    continue
            return annots, annots_file_names

    def __load_txt_annots(self, annot_path: str, is_dir: bool):
        annots = []
        has_header = False
        header = []
        annots_file_names = []
        if not is_dir:
            with open(annot_path, "r") as f:
                annot_reader = f.readlines()
            for index, line in enumerate(annot_reader):
                line = self.__str_to_list(line)
                if index == 0:
                    if self.__is_header(line):
                        has_header = True
                        header = line
                if has_header:
                    annot_dict = {}
                    for key, value in zip(header, line):
                        annot_dict[key] = value
                    annots.append(annot_dict)
                else:
                    annots.append(line)
            return annots, None
        else:
            for file in os.listdir(annot_path):
                if file.endswith(".txt"):
                    with open(f"{annot_path}/{file}", "r") as f:
                        annot_reader = f.readlines()
                    for index, line in enumerate(annot_reader):
                        line = self.__str_to_list(line)
                        if index == 0:
                            if self.__is_header(line):
                                has_header = True
                                header = line
                                continue
                        if has_header:
                            annot_dict = {}
                            for key, value in zip(header, line):
                                annot_dict[key] = value
                            annots.append(annot_dict)
                        else:
                            annots.append(line)
                            annots_file_names.append(file.removesuffix(".txt"))
                else:
                    continue
            return annots, annots_file_names

    def __list_configs(self):
        unpacked = {}
        for key in self.advanced_configs:
            if self.advanced_configs[key]:
                unpacked[key] = self.__str_to_list(self.advanced_configs[key])
            else:
                unpacked[key] = []
        self.advanced_configs = unpacked

    @staticmethod
    def __list_product(_list: list):
        if not _list:
            return 0
        product = _list[0]
        for index in range(1, len(_list)):
            product = product * _list[index]
        return product

    def __sort_annots(self):
        return sorted(self.annots, key=itemgetter("image_id"))

    def __group_annots(self):
        group = None
        grouped = []
        group_dict = {}
        for index, annot in enumerate(self.annots):
            if group != annot["image_id"]:
                if group_dict:
                    grouped.append(group_dict)
                group_dict = {}
                group = annot["image_id"]
                group_dict["image"] = group
                group_dict["annotations"] = []
            group_dict["annotations"].append(annot)
        grouped.append(group_dict)
        return grouped

    def __get_index_item(self, item, index: str, index_list: list, current_list_index: int):
        if index == ".n":
            if type(item) != list:
                if index_list[current_list_index] == 0:
                    index_item = item
                    list_index = current_list_index + 1
                    continuation = "continue"
                else:
                    index_item = None
                    list_index = current_list_index
                    continuation = "return"
            else:
                if len(item) > index_list[current_list_index]:
                    index_item = item[index_list[current_list_index]]
                    list_index = current_list_index + 1
                    continuation = "continue"
                else:
                    index_item = None
                    list_index = current_list_index
                    continuation = "return"
        elif index == ".name":
            index_item = self.annots_file_names[index_list[0]]
            list_index = current_list_index
            continuation = "break"
        elif index.startswith("."):
            index = index.removeprefix(".")
            if index.endswith("key"):
                keys = item.keys()
                index_item = keys[int(index.removesuffix("key"))]
                list_index = current_list_index
                continuation = "break"
            elif index.isdigit():
                # can also iterate through dicts, mostly due to inconsistencies with csv files
                if type(item) == dict:
                    keys = item.keys()
                    index_item = keys[int(index)]
                    list_index = current_list_index
                    continuation = "continue"
                elif type(item) == list:
                    index_item = item[int(index)]
                    list_index = current_list_index
                    continuation = "continue"
                else:
                    raise TypeError(f"Expected type list, found {type(item)} for index {index}.")
            else:
                raise ValueError(f"Unknown index: {index}.")
        else:
            index_item = item[index]
            list_index = current_list_index
            continuation = "continue"
        return index_item, list_index, continuation

    @staticmethod
    def __is_header(item_list: list):
        check_list = []
        for item in item_list:
            # replace '.', because otherwise floats were not recognized as digits
            check_list.append(item.replace(".", "").isdigit())
        return not any(check_list)

    @staticmethod
    def __str_to_list(string: str) -> list:
        if "," in string:
            _list = string.replace(" ", "").split(",")
        elif "/" in string:
            _list = string.replace(" ", "").split("/")
        elif "|" in string:
            _list = string.replace(" ", "").split("|")
        else:
            _list = string.split(" ")
        return _list

    def __validate_configs(self):
        is_dir = self.basic_configs["is_dir"]
        for index, key in enumerate(self.advanced_configs.keys()):
            config = self.advanced_configs[key]
            if not config:
                continue
            if is_dir and config[0] != ".n" and config[0] != ".name":
                config.reverse()
                config.append(".n")
                config.reverse()
                self.advanced_configs[key] = config
                print("INFO: Added iterator to config")
            else:
                continue


class ImageAdaptor:
    """
    Adaptor for loading image paths and names.
    Adaptor takes a config file.
    Unlike the AnnotationAdaptor, ImageAdaptor can not be configured. Images names are always represented by the file
    name without the extension e.g.: 127532.jpg -> name = 127532
    """

    def __init__(self, configs: dict):
        self.basic_configs = configs["basic"]
        self.accepted_dtypes = ["jpg", "png", "jpeg"]

    def load_imgs(self):
        img_dir = self.basic_configs["img_path"]
        imgs = {}
        for item in os.listdir(img_dir):
            img_extension = pathlib.Path(item).suffix.removeprefix(".")
            if img_extension in self.accepted_dtypes:
                img_path = f"{img_dir}/{item}"
                img_width, img_height = imagesize.get(img_path)
                img_name = item.removesuffix(f".{img_extension}")
                img_uses = 1
                imgs[img_name] = {"name": img_name,
                                  "path": img_path,
                                  "uses": img_uses,
                                  "width": img_width,
                                  "height": img_height}
        return imgs


def load_dataset(configs: dict):
    img_adaptor = ImageAdaptor(configs)
    imgs = img_adaptor.load_imgs()
    annot_adaptor = AnnotationAdaptor(configs)
    annots = annot_adaptor.load_annots()
    data_container = []
    for annot in annots:
        if annot["image"] not in imgs.keys():
            raise ValueError(f"Found image in annotations, that is not part of images: {annot['image']}.")
        img_name = annot["image"]
        annot["image"] = imgs[img_name]
        data_container.append(annot)
        imgs.pop(img_name)
    if len(imgs.keys()) > 0:
        print(f"Found {len(imgs.keys())} remaining images with no annotations. Added as empty images.")
        for key in imgs.keys():
            data_container.append({"image": imgs[key], "annotations": []})
    return data_container
