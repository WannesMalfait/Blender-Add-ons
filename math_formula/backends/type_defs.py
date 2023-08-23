from enum import IntEnum, auto
from dataclasses import dataclass
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


# Penalty for converting from type a to type b
# Higher means worse; 1000 means never do this conversion
dtype_conversion_penalties = (
    (0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (0, 0, 0, 11, 12, 14, 13, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 22, 0, 11, 13, 12, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 23, 22, 0, 12, 11, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 25, 24, 23, 0, 21, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 24, 23, 22, 11, 0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 0,
     1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     0, 1000, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 0, 1000, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 0, 1000, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 0, 1000, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 0, 1000, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000, 0, 1000, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000, 1000, 0, 1000),
    (0, 0, 1000, 1000, 1000, 1000, 1000, 1000,
     1000, 1000, 1000, 1000, 1000, 1000, 1000, 0),
)
assert DataType.ROTATION.value == len(
    dtype_conversion_penalties)-1, 'Correct table size'
assert DataType.ROTATION.value == len(
    dtype_conversion_penalties[0])-1, 'Correct table size'

string_to_data_type = {
    '_': DataType.UNKNOWN,
    '': DataType.DEFAULT,
    'bool': DataType.BOOL,
    'int': DataType.INT,
    'float': DataType.FLOAT,
    'rgba': DataType.RGBA,
    'vec3': DataType.VEC3,
    'geo': DataType.GEOMETRY,
    'str': DataType.STRING,
    'shader': DataType.SHADER,
    'obj': DataType.OBJECT,
    'img': DataType.IMAGE,
    'collection': DataType.COLLECTION,
    'tex': DataType.TEXTURE,
    'mat': DataType.MATERIAL,
    'quaternion': DataType.ROTATION,
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


class Operation():
    def __init__(self, op_type: OpType, data) -> None:
        self.op_type = op_type
        self.data = data

    def __str__(self) -> str:
        return f"({self.op_type.name}, {self.data})"

    def __repr__(self) -> str:
        return self.__str__()


class FileData():
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
class CompiledFunction():
    inputs: list[str]
    body: list[Operation]
    num_outputs: int


@dataclass
class CompiledNodeGroup():
    name: str
    inputs: list[TyArg]
    outputs: list[TyArg]
    body: list[Operation]
