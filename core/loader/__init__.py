"""
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

How should a loading query look like?
    - [] or {} are used to specify when to load from a list or a dictionary.
    - [n] indicates the iteration of a list without a specified index
    - [1] indicates the list entry at index 1
    - /filename/ reads the name of the file
    The query for x_min in the COCO dataset would be: '{annotations}[n]{bbox}[0]'
    The query for x_min of a typical txt-file inside a folder would be: '[n][1]'
    The query to get the name of the file would be: '/filename/'
"""
