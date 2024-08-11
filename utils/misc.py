"""
Collection of unreferrable functions.
"""

import uuid
import random
from typing import List, Tuple
import json

import numpy as np
from cv2 import imread
import imagesize


def is_in_dict(key: str, dict_: dict):
    try:
        _ = dict_[key]
        return True
    except KeyError:
        return False


def new_id() -> str:
    return str(uuid.uuid4())


def get_random() -> float:
    return random.random()


def get_seed() -> int:
    return int(get_random() * 90000) + 10000


def set_seed(seed: int) -> None:
    random.seed = seed


def is_executed(execution_probability: float) -> bool:
    return True if get_random() < execution_probability else False


def choose_by_prob(value_list: list, probability_list: List[float]):
    rand = get_random()
    assert len(value_list) == len(probability_list), "Ambiguous value to probability relation."
    prob_sum = 0
    for value, prob in zip(value_list, probability_list):
        prob_sum += prob
        if rand < prob_sum:
            return value
    else:
        raise NotImplementedError("Something went wrong here, this code should be unreachable.")


def load_json(file_path) -> dict:
    """
    Loads a json file and stores it as dict
    """
    with open(file_path, "r") as f:
        json_dict = json.load(f)
    return json_dict


def read_img(img_path: str) -> np.ndarray:
    return np.einsum("ijk->jik", imread(img_path))


def is_header(item_list: List[str]) -> bool:
    """
    Checks if a row is a header row.
    Args:
        item_list (List[str]): List of items in row to check
    """
    return not any([item.replace(".", "").isdigit() for item in item_list])


def string_to_list(string: str) -> list:
    """
    Splits a string into a list. Takes a space as separator.
    """
    return string.split(" ")


def img_dims(path: str) -> Tuple[int, int]:
    """
   Gets image width and height.
    Args:
        path (str): Path to image as string
    Returns:
        (Tuple[int, int]): image width, image height
    """
    return imagesize.get(path)


def list_intersection(a: list, b: list) -> list:
    """
    Returns a List with all elements that are equal in list a and list b.
    Args:
        a (list): List a
        b (list): List b
    Returns:
        (list): Intersection list of a and b
    """
    return list(set(a).intersection(set(b)))


def is_api_key(string: str):
    if not len(string) == 64:
        return False
    if string.find("/") != -1:
        return False
    if string.find("\\") != -1:
        return False
    return True


def get_config_from_api(api_key: str):
    pass




































