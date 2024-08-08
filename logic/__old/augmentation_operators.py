
class BaseOperator:
    def __init__(self):
        """
        An Operator Node connects paths.
        """
        self.source = None
        self.output = None
        self.depends = None


class DatasetOperator(BaseOperator):
    pass


class FilterOperator(BaseOperator):
    pass


class MergeOperator(BaseOperator):
    pass


class SplitOperator(BaseOperator):
    pass


class DuplicateOperator(BaseOperator):
    pass


class ModifyOperator(BaseOperator):
    pass
