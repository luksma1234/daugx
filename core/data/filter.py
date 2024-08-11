from typing import Callable, List, Union, Tuple

from daugx.core import constants as c
from daugx.core.data.data import DataPackage, Dataset


class Filter:
    def __init__(self, type_: str, specifier: dict, operator: str, value: Union[int, float, None]):
        self.__type = type_
        self.__specifier = specifier
        self.__specifier_value = self.__specifier[c.FILTER_SPECIFIER_VALUE]
        self.__specifier_category = self.__specifier[c.FILTER_SPECIFIER_CATEGORY]
        self.__operator = operator
        self.__value = value

        # __comparator is compared to __value
        self.__comparator = None

    def is_filtered(self, data_package: DataPackage) -> bool:
        """
        Matches the filter type and parses resulting data to specifier.
        Filter result is propagated back upwards.
        Args:
            data_package (DataPackage): Any DataPackage
        Returns:
            (bool): If data_package matches filter criteria
        """
        meta_inf = data_package.meta_inf
        match self.__type:
            case c.FILTER_TYPE_MIN_AREA:
                return self._match_specifier(
                    meta_inf.min_area_by_label_name,
                    meta_inf.min_area_by_label_id,
                    meta_inf.min_area,
                )
            case c.FILTER_TYPE_MAX_AREA:
                return self._match_specifier(
                    meta_inf.max_area_by_label_name,
                    meta_inf.max_area_by_label_id,
                    meta_inf.max_area,
                )
            case c.FILTER_TYPE_MIN_WIDTH:
                return self._match_specifier(
                    meta_inf.min_width_by_label_name,
                    meta_inf.min_width_by_label_id,
                    meta_inf.min_width,
                )
            case c.FILTER_TYPE_MAX_WIDTH:
                return self._match_specifier(
                    meta_inf.max_width_by_label_name,
                    meta_inf.max_width_by_label_id,
                    meta_inf.max_width,
                )
            case c.FILTER_TYPE_MIN_HEIGHT:
                return self._match_specifier(
                    meta_inf.min_height_by_label_name,
                    meta_inf.min_height_by_label_id,
                    meta_inf.min_height,
                )
            case c.FILTER_TYPE_MAX_HEIGHT:
                return self._match_specifier(
                    meta_inf.max_height_by_label_name,
                    meta_inf.max_height_by_label_id,
                    meta_inf.max_height,
                )
            case c.FILTER_TYPE_LABEL:
                return self._match_specifier(
                    meta_inf.label_names,
                    meta_inf.label_ids,
                    meta_inf.annotations.annots,
                )
            case c.FILTER_TYPE_NLABEL:
                return self._match_specifier(
                    len(meta_inf.label_names),
                    len(meta_inf.label_ids),
                    len(meta_inf.annotations.annots),
                )

    def _match_specifier(
            self,
            data_if_name: Union[Callable, list, int],
            data_if_id: Union[Callable, list, int],
            data_if_any: Union[Callable, list, int]
    ) -> bool:
        """
        Matches specifier function. Parses result to _match_operator for final comparison.
        """
        match self.__specifier_category:
            case c.FILTER_SPECIFIER_CATEGORY_NAME:
                if isinstance(data_if_name, list) or isinstance(data_if_name, int):
                    self.__comparator = data_if_name
                else:
                    self.__comparator = data_if_name(self.__specifier_value)
            case c.FILTER_SPECIFIER_CATEGORY_ID:
                if isinstance(data_if_id, list) or isinstance(data_if_name, int):
                    self.__comparator = data_if_id
                else:
                    self.__comparator = data_if_id(self.__specifier_value)
            case c.FILTER_SPECIFIER_CATEGORY_ANY:
                if isinstance(data_if_any, list) or isinstance(data_if_name, int):
                    self.__comparator = data_if_any
                else:
                    self.__comparator = data_if_any()
        return self._match_operator()

    def _match_operator(self) -> bool:
        """
        Matches the compartor to the value depending on the operator.
        """
        if self.__comparator is None:
            return True
        match self.__operator:
            case c.FILTER_OPERATOR_GREATER_THAN:
                return self.__comparator > self.__value
            case c.FILTER_OPERATOR_SMALLER_THAN:
                return self.__comparator < self.__value
            case c.FILTER_OPERATOR_EQUALS_OR_GREATER_THAN:
                return self.__comparator >= self.__value
            case c.FILTER_OPERATOR_EQUALS_OR_SMALLER_THAN:
                return self.__comparator <= self.__value
            case c.FILTER_OPERATOR_EQUALS:
                return self.__comparator == self.__value
            case c.FILTER_OPERATOR_EXISTS:
                if not isinstance(self.__comparator, list):
                    return True
                if self.__specifier_category != c.FILTER_SPECIFIER_CATEGORY_ANY:
                    return self.__specifier_value in self.__comparator
                else:
                    return len(self.__comparator) > 0
            case c.FILTER_OPERATOR_NOT_EXISTS:
                if not isinstance(self.__comparator, list):
                    return True
                if self.__specifier_category != c.FILTER_SPECIFIER_CATEGORY_ANY:
                    return self.__specifier_value not in self.__comparator
                else:
                    return len(self.__comparator) == 0


class FilterSequence:
    def __init__(self, id_: str, is_revered: bool = False):
        self.__id = id_
        self.__is_reversed = is_revered
        self.__sequence: List[tuple] = []
        self.__res: bool = True
        self.__included = []
        self.__excluded = []

    @property
    def id(self):
        return self.__id

    @property
    def is_reversed(self):
        return self.__is_reversed

    def add(self, filter_: Filter, chain_operator: str):
        """
        Adds a filter and an operator to the sequence. The Operator can be one of ['AND', 'OR', 'NONE']
        """
        self.__sequence.append((filter_, chain_operator))

    def filter(self, data_packages: List[DataPackage]) -> List[int]:
        """
        Filters a Dataset. Runs every data package through all filters and checks if the package matches all filter
        criteria.
        Args:
            data_packages (Dataset): Any Dataset
        Returns:
            (List[int]): Depending on the parameter is_reversed:
                         True: A list of indexes which match the filter criteria
                         False: A list of indexes which do not match the filter criteria
        """
        self._reset()
        for index, data_package in enumerate(data_packages):
            self._execute(data_package)
            if self.__res:
                self.__included.append(index)
            else:
                self.__excluded.append(index)
            self.__res = True
        if self.__is_reversed:
            return self.__excluded
        else:
            return self.__included

    def _execute(self, data_package):
        for filter_, operator in self.__sequence:
            self.__res = filter_.is_filtered(data_package)
            if not self._is_continued(operator):
                break

    def _reset(self):
        self.__res = True
        self.__included = []
        self.__excluded = []

    def _is_continued(self, operator: str) -> bool:
        match operator:
            case c.FILTER_SEQUENCE_OPERATOR_OR:
                if self.__res:
                    return False
            case c.FILTER_SEQUENCE_OPERATOR_AND:
                if not self.__res:
                    return False
            case c.FILTER_SEQUENCE_OPERATOR_NONE:
                return False
        return True
