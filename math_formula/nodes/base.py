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

    def can_convert(self, other: 'DataType') -> bool:
        if self == DataType.UNKNOWN:
            return True
        elif self == other:
            return True
        elif DataType.GEOMETRY <= self.value <= DataType.STRING or DataType.GEOMETRY <= other.value <= DataType.STRING:
            return False
        return True


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
vec3 = list[float]

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

    def convert(self, to_type: DataType) -> ValueType:
        assert self.data_type.can_convert(
            to_type), 'Only convert when possible'
        if self.data_type == DataType.BOOL:
            if to_type == DataType.INT:
                self.value = int(self.value)
            if to_type == DataType.FLOAT:
                self.value = float(self.value)
            if to_type == DataType.RGBA:
                self.value = [float(self.value) for _ in range(4)]
            if to_type == DataType.VEC3:
                self.value = [float(self.value) for _ in range(3)]
        if self.data_type == DataType.INT:
            if to_type == DataType.BOOL:
                self.value = bool(self.value <= 0)
            if to_type == DataType.FLOAT:
                self.value = float(self.value)
            if to_type == DataType.RGBA:
                self.value = [float(self.value) for _ in range(4)]
            if to_type == DataType.VEC3:
                self.value = [float(self.value) for _ in range(3)]
        if self.data_type == DataType.FLOAT:
            if to_type == DataType.BOOL:
                self.value = bool(self.value <= 0.0)
            if to_type == DataType.INT:
                self.value = int(self.value)
            if to_type == DataType.RGBA:
                self.value = [self.value for _ in range(4)]
            if to_type == DataType.VEC3:
                self.value = [self.value for _ in range(3)]
        if self.data_type == DataType.RGBA:
            gray_scale = (
                0.2126 * self.value[0]) + (0.7152 * self.value[1]) + (0.0722 * self.value[2])
            if to_type == DataType.BOOL:
                self.value = bool(gray_scale)
            if to_type == DataType.INT:
                self.value = int(gray_scale)
            if to_type == DataType.FLOAT:
                self.value = gray_scale
            if to_type == DataType.VEC3:
                self.value = [self.value[i] for i in range(3)]
        if self.data_type == DataType.VEC3:
            avg = (
                self.value[0] + self.value[1] + self.value[2])/3.0
            if to_type == DataType.BOOL:
                self.value = bool(avg)
            if to_type == DataType.INT:
                self.value = int(avg)
            if to_type == DataType.FLOAT:
                self.value = avg
            if to_type == DataType.RGBA:
                self.value = self.value + [1]
        self.data_type = to_type
        return self.value


Struct = Tuple[Value]


class Socket():
    def __init__(self, index: int, name: str, sock_type: DataType) -> None:
        self.index = index
        self.name = name
        self.sock_type = sock_type

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Socket[{self.index}]: {self.name}: {data_type_to_string[self.sock_type]}'


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

    def __str__(self) -> str:
        return f'function: {self._name}[{self.prop_values}]'

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
