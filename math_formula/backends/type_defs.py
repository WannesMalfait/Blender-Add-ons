from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Union


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
    SHADER = auto()
    OBJECT = auto()
    IMAGE = auto()
    COLLECTION = auto()
    TEXTURE = auto()
    MATERIAL = auto()
    ROTATION = auto()
    MATRIX = auto()
    MENU = auto()


# Penalty for converting from type a to type b
# Higher means worse; 1000 means never do this conversion
# Default to disallowing conversions.
dtype_conversion_penalties = [
    [1000 if type_a != type_b else 0 for type_b in DataType] for type_a in DataType
]

dtype_conversion_penalties[DataType.BOOL.value][DataType.INT.value] = 11
dtype_conversion_penalties[DataType.BOOL.value][DataType.FLOAT.value] = 12
dtype_conversion_penalties[DataType.BOOL.value][DataType.VEC3.value] = 13
dtype_conversion_penalties[DataType.BOOL.value][DataType.RGBA.value] = 14

dtype_conversion_penalties[DataType.INT.value][DataType.BOOL.value] = 22
dtype_conversion_penalties[DataType.INT.value][DataType.FLOAT.value] = 11
dtype_conversion_penalties[DataType.INT.value][DataType.VEC3.value] = 12
dtype_conversion_penalties[DataType.INT.value][DataType.RGBA.value] = 13

dtype_conversion_penalties[DataType.FLOAT.value][DataType.BOOL.value] = 23
dtype_conversion_penalties[DataType.FLOAT.value][DataType.INT.value] = 22
dtype_conversion_penalties[DataType.FLOAT.value][DataType.VEC3.value] = 11
dtype_conversion_penalties[DataType.FLOAT.value][DataType.RGBA.value] = 12

dtype_conversion_penalties[DataType.VEC3.value][DataType.BOOL.value] = 24
dtype_conversion_penalties[DataType.VEC3.value][DataType.INT.value] = 23
dtype_conversion_penalties[DataType.VEC3.value][DataType.FLOAT.value] = 22
dtype_conversion_penalties[DataType.VEC3.value][DataType.RGBA.value] = 11

dtype_conversion_penalties[DataType.RGBA.value][DataType.BOOL.value] = 25
dtype_conversion_penalties[DataType.RGBA.value][DataType.INT.value] = 24
dtype_conversion_penalties[DataType.RGBA.value][DataType.FLOAT.value] = 23
dtype_conversion_penalties[DataType.RGBA.value][DataType.VEC3.value] = 21

for dtype in DataType:
    dtype_conversion_penalties[DataType.UNKNOWN.value][dtype.value] = 1
    dtype_conversion_penalties[DataType.DEFAULT.value][dtype.value] = 1
    dtype_conversion_penalties[dtype.value][DataType.UNKNOWN.value] = 0
    dtype_conversion_penalties[dtype.value][DataType.DEFAULT.value] = 0


string_to_data_type = {
    "_": DataType.UNKNOWN,
    "": DataType.DEFAULT,
    "bool": DataType.BOOL,
    "int": DataType.INT,
    "float": DataType.FLOAT,
    "rgba": DataType.RGBA,
    "vec3": DataType.VEC3,
    "geo": DataType.GEOMETRY,
    "str": DataType.STRING,
    "shader": DataType.SHADER,
    "obj": DataType.OBJECT,
    "img": DataType.IMAGE,
    "collection": DataType.COLLECTION,
    "tex": DataType.TEXTURE,
    "material": DataType.MATERIAL,
    "quaternion": DataType.ROTATION,
    "matrix": DataType.MATRIX,
    "menu": DataType.MENU,
}

ValueType = Union[bool, int, float, list[float], str, None]


class OpType(IntEnum):
    # Push the given value on the stack. None represents a default value.
    PUSH_VALUE = 0
    # Create a variable with the given name, and assign it to stack.pop().
    CREATE_VAR = auto()
    # Get the variable with the given name, and push it onto the stack.
    GET_VAR = auto()
    # Replace the last item on the stack with the element at the given index.
    # The indexing is reversed.
    # If stack looked like [x,y,[z,w]] then after GET_OUTPUT 1 it looks like
    # [x,y,z]
    GET_OUTPUT = auto()
    # Set the ouput of the last added node to the given value. Data is a
    # tuple of the output index and the value to be set.
    SET_OUTPUT = auto()
    # Set the functions output at the given index to the value on top of the stack.
    SET_FUNCTION_OUT = auto()
    # Split the last item on the stack into individual items. So if the
    # stack looked like [x,y,[z,w,v]] it would become [x,y,z,w,v].
    SPLIT_STRUCT = auto()
    # Call the given function, all the arguments are on the stack. The data
    # is a CompiledFunction
    CALL_FUNCTION = auto()
    # Same as CALL_FUNCTION but a node group is created.
    CALL_NODEGROUP = auto()
    # Create the built-in node. Data is a NodeInstance.
    CALL_BUILTIN = auto()
    # Set the label of the last added node to the given name.
    RENAME_NODE = auto()
    # Clear the stack.
    END_OF_STATEMENT = auto()


@dataclass
class BuiltinNode:
    # The input sockets, name and data type
    inputs: list[tuple[str, DataType]]
    # The output sockets, name and data type
    outputs: list[tuple[str, DataType]]


@dataclass
class NodeInstance:
    # Key in the nodes dict
    key: str
    # Input indices that are used
    inputs: list[int]
    # Output indices that are used
    outputs: list[int]
    # Props that are set along with their value
    props: list[tuple[str, ValueType]]


class Operation:
    def __init__(self, op_type: OpType, data) -> None:
        self.op_type = op_type
        self.data = data

    def __str__(self) -> str:
        return f"({self.op_type.name}, {self.data})"

    def __repr__(self) -> str:
        return self.__str__()


class FileData:
    def __init__(self) -> None:
        self.geometry_nodes: dict[str, list[TyFunction]] = {}
        self.shader_nodes: dict[str, list[TyFunction]] = {}

    def num_funcs(self) -> int:
        tot = 0
        for value in self.geometry_nodes.values():
            tot += len(value)
        for value in self.shader_nodes.values():
            tot += len(value)
        return tot


class StackType(IntEnum):
    VALUE = 0
    SOCKET = auto()
    STRUCT = auto()
    EMPTY = auto()


class ty_ast:
    ...


class ty_stmt(ty_ast):
    ...


@dataclass
class TyArg(ty_ast):
    name: str
    dtype: DataType
    value: Union[None, ValueType]


@dataclass
class TyFunction(ty_ast):
    inputs: list[TyArg]
    outputs: list[TyArg]
    body: list[ty_stmt]
    used_outputs: list[bool]
    is_node_group: bool
    name: str


@dataclass
class TyRepr(ty_ast):
    body: list[ty_stmt]


@dataclass
class ty_expr(ty_stmt):
    stype: StackType
    dtype: list[DataType]
    out_names: list[str]


@dataclass
class Const(ty_expr):
    value: ValueType


@dataclass
class Var(ty_expr):
    id: str
    needs_instantion: bool


@dataclass
class NodeCall(ty_expr):
    node: NodeInstance
    args: list[ty_expr]


@dataclass
class FunctionCall(ty_expr):
    function: TyFunction
    args: list[ty_expr]


@dataclass
class GetOutput(ty_expr):
    value: ty_expr
    index: int


@dataclass
class TyAssign(ty_stmt):
    targets: list[Union[Var, None]]
    value: ty_expr


@dataclass
class TyOut(ty_stmt):
    targets: list[Union[int, None]]
    value: ty_expr


@dataclass
class TyLoop(ty_stmt):
    var: Union[None, Var]
    start: int
    end: int
    body: list[ty_stmt]


@dataclass
class CompiledFunction:
    inputs: list[str]
    body: list[Operation]
    num_outputs: int


@dataclass
class CompiledNodeGroup:
    name: str
    inputs: list[TyArg]
    outputs: list[TyArg]
    body: list[Operation]
