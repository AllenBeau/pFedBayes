import keyword


class ArgumentParser:
    """
    Argument mapping parser.
    """
    def __init__(self, description=None):
        self.description = description
        self.__data = {}    # delegate the functionality to dict.

    def __getattr__(self, name):
        """
        Dynamic attribute method.
        :param name: attribute name.
        :return: attribute value.
        """
        if hasattr(self.__data, name):
            return getattr(self.__data, name)
        else:
            return self.__data[name]

    def add_argument(self, name, type=None, default=None, **kwargs):
        name = name.replace('--', '')
        # the name should be a valid keyword.
        if keyword.iskeyword(name) or not name.isidentifier():
            name += '_'
        self.__data[name] = default if type is None else type(default)

    def parse_args(self):
        return self