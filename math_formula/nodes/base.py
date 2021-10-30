from typing import List, Union, Tuple
from enum import IntEnum, auto
from bpy.types import NodeSocket


class DataType(IntEnum):
    UNKNOWN = 0
    DEFAULT = auto()
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    RGBA = auto()
    VEC3 = auto()
    GEOMETRY = auto()
    STRING = auto()


string_to_data_type = {
    '_': DataType.UNKNOWN,
    '': DataType.DEFAULT,
    'bool': DataType.BOOL,
    'int': DataType.INT,
    'float': DataType.FLOAT,
    'rgba': DataType.RGBA,
    'vec3': DataType.VEC3,
    'geometry': DataType.GEOMETRY,
    'string': DataType.STRING,
}

data_type_to_string = {value: key for key,
                       value in string_to_data_type.items()}
vec3 = tuple[float]

ValueType = Union[None, bool, str, int, float,
                  vec3, NodeSocket]


class Value():
    def __init__(self,
                 data_type: DataType,
                 value: ValueType) -> None:
        self.data_type = data_type
        self.value = value

    def __str__(self) -> str:
        return f"{self.value} ({data_type_to_string[self.data_type]})"

    def __repr__(self) -> str:
        return self.__str__()


Struct = Tuple[Value]


class Socket():
    def __init__(self, index: int, name: str, sock_type: DataType) -> None:
        self.index = index
        self.name = name
        self.sock_type = sock_type

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Socket[{self.index}]: {self.name}: {self.sock_type}'


class NodeFunction():
    _name = ''
    _input_sockets: list[Socket] = []
    _output_sockets: list[Socket] = []
    _props = {}

    def __init__(self, props) -> None:
        self._input_sockets = type(self)._input_sockets
        self._output_sockets = type(self)._output_sockets
        self.prop_values = []

    def input_sockets(self) -> list[Socket]:
        return self._input_sockets

    def output_sockets(self) -> list[Socket]:
        return self._output_sockets

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def invalid_prop_values(cls, props) -> list[str]:
        """
        Returns the property values that are invalid.
        The number of props should be less than or equal
        to the number of `cls._props`.
        """
        invalid = []
        for i, prop in enumerate(props):
            if prop is None:
                continue
            if prop in cls._props[i][1]:
                continue
            invalid.append(prop)
        return invalid

    def props(self) -> list[tuple[str, Union[str, None]]]:
        return self.prop_values
