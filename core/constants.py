

# Annotation constants

# Annotation query constants

# Annotation query keywords
QUERY_LABEL_NAME = "LABELNAME"
QUERY_LABEL_ID = "LABELID"
QUERY_IMAGE_REF = "IMAGEREF"
QUERY_X_MIN = "XMIN"
QUERY_Y_MIN = "YMIN"
QUERY_X_MAX = "XMAX"
QUERY_Y_MAX = "YMAX"
QUERY_X_CENTER = "XCENTER"
QUERY_Y_CENTER = "YCENTER"
QUERY_WIDTH = "WIDTH"
QUERY_HEIGHT = "HEIGHT"
QUERY_POLYGON = "POLYGON"
QUERY_KEYPOINT = "KEYPOINT"
QUERY_CUSTOM = "CUSTOM"

# Annotation query loading query constants

QUERY_UNDEFINED_ITERATOR = "[n]"
QUERY_CURRENT_FILE_NAME = "/filename/"

# Annotation query mode constants

QUERY_MODE_ONE_FILE = "onefile"
QUERY_MODE_DIRECTORY = "directory"

# Annotation query lists

QUERY_KEYWORDS_LIST = [
    QUERY_X_MIN, QUERY_Y_MIN, QUERY_X_MAX, QUERY_Y_MAX, QUERY_WIDTH, QUERY_HEIGHT, QUERY_X_CENTER, QUERY_Y_CENTER,
    QUERY_KEYPOINT, QUERY_POLYGON, QUERY_IMAGE_REF, QUERY_LABEL_NAME, QUERY_LABEL_ID, QUERY_CUSTOM
]
QUERY_BBOX_KEYWORDS_LIST = [
    QUERY_X_MIN, QUERY_Y_MIN, QUERY_X_MAX, QUERY_Y_MAX, QUERY_WIDTH, QUERY_HEIGHT, QUERY_X_CENTER, QUERY_Y_CENTER
]
QUERY_KEYPOINT_KEYWORDS_LIST = [QUERY_KEYPOINT]
QUERY_POLYGON_KEYWORDS_LIST = [QUERY_POLYGON]
QUERY_MANDATORY_KEYWORDS_LIST = [QUERY_IMAGE_REF]
QUERY_OPTIONAL_KEYWORDS_LIST = [QUERY_LABEL_NAME, QUERY_LABEL_ID, QUERY_CUSTOM]

# Annotation boundary types
BOUNDARY_TYPE_BBOX = "bbox"
BOUNDARY_TYPE_KEYPOINT = "keypoint"
BOUNDARY_TYPE_POLYGON = "polygon"


# Annotation raw dictionary keys
DICTIONARY_KEY_BOUNDARY_POINTS = "boundary_points"
DICTIONARY_KEY_LABEL_ID = "label_id"
DICTIONARY_KEY_LABEL_NAME = "label_name"
DICTIONARY_KEY_IMAGE_REF = "image_ref"

# BBox valid keyword pairs
KEYWORD_PAIR_XY_MIN = "xy_min"
KEYWORD_PAIR_XY_MAX = "xy_max"
KEYWORD_PAIR_CENTER = "center"
KEYWORD_PAIR_WH = "wh"


# Regex patterns

REGEX_QUERY_UNDEFINED_ITERATOR = "\[n]"

# Filter constants

FILTER_TYPE_MIN_AREA = "MinArea"
FILTER_TYPE_MAX_AREA = "MaxArea"
FILTER_TYPE_MIN_WIDTH = "MinWidth"
FILTER_TYPE_MAX_WIDTH = "MaxWidth"
FILTER_TYPE_MIN_HEIGHT = "MinHeight"
FILTER_TYPE_MAX_HEIGHT = "MaxHeight"
FILTER_TYPE_LABEL = "Label"
FILTER_TYPE_NLABEL = "NLabel"

FILTER_SPECIFIER_CATEGORY = "category"
FILTER_SPECIFIER_CATEGORY_NAME = "name"
FILTER_SPECIFIER_CATEGORY_ID = "id"
FILTER_SPECIFIER_CATEGORY_ANY = "any"
FILTER_SPECIFIER_VALUE = "value"

FILTER_OPERATOR_GREATER_THAN = ">"
FILTER_OPERATOR_SMALLER_THAN = "<"
FILTER_OPERATOR_EQUALS = "="
FILTER_OPERATOR_EQUALS_OR_GREATER_THAN = ">="
FILTER_OPERATOR_EQUALS_OR_SMALLER_THAN = "<="
FILTER_OPERATOR_EXISTS = "true"
FILTER_OPERATOR_NOT_EXISTS = "false"

FILTER_SEQUENCE_OPERATOR = "operator"
FILTER_SEQUENCE_FILTER = "filter"
FILTER_SEQUENCE_OPERATOR_AND = "AND"
FILTER_SEQUENCE_OPERATOR_OR = "OR"
FILTER_SEQUENCE_OPERATOR_NONE = "NONE"

FILTER_DICT_ID = "id"
FILTER_DICT_SEQUENCE = "sequence"
FILTER_DICT_IS_REVERSED = "is_reversed"
FILTER_DICT_TYPE = "type"
FILTER_DICT_SPECIFIER = "specifier"
FILTER_DICT_OPERATOR = "operator"
FILTER_DICT_VALUE = "value"
FILTER_DICT_CHAIN_OPERATOR = "chain_operator"

# General Constants
NODE_TYPE_STR = "type"
NODE_NEXT_STR = "next"
NODE_PARAMS_STR = "params"

# Node Constants
NODE_STR = "node"
NODES_STR = "nodes"

NODE_TYPE_INPUT = "input"
NODE_TYPE_OUTPUT = "output"
NODE_TYPE_MERGE = "merge"
NODE_TYPE_SPLIT = "split"
NODE_TYPE_FILTER = "filter"
NODE_TYPE_AUGMENTATION = "augmentation"

# config constants

CONFIG_KEY_WORKFLOW = "workflow"
CONFIG_KEY_DATASETS = "datasets"
CONFIG_KEY_INIT = "init"
CONFIG_KEY_FILTER = "filter"
CONFIG_KEY_ID = "id"














