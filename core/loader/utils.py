import json


def dict_to_json(dict_object):
    return json.dumps(dict_object, indent=4)


def pt_to_ls(pt: str) -> list:
    """
    Converts current pointer self.cur_pt to list.
    :return: list of pointer items separated by "," or " " or "|" or "/"
    """
    if "," in pt:
        _list = pt.replace(" ", "").split(",")
    elif "/" in pt:
        _list = pt.replace(" ", "").split("/")
    elif "|" in pt:
        _list = pt.replace(" ", "").split("|")
    else:
        _list = pt.split(" ")
    return _list


def extr_int(str_: str) -> int:
    """
    Extracts one integer of any string.
    :param str_: Any string
    :return: integer of string or 0 if no integer in string
    """
    digits = ""
    for char in str_:
        if char.isdigit():
            digits += char
    if digits:
        return int(digits)
    return 0
