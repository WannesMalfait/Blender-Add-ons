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


# Penalty for converting from type a to type b
# Higher means worse; 100 means never do this conversion
dtype_conversion_penalties = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 1, 2, 4, 3, 100, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 12, 0, 1, 3, 2, 100, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 13, 12, 0, 2, 1, 100, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 15, 14, 13, 0, 11, 100, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 14, 13, 12, 1, 0, 100, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0, 100),
    (0, 0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 0),
)
assert DataType.MATERIAL.value == len(
    dtype_conversion_penalties)-1, 'Correct table size'
assert DataType.MATERIAL.value == len(
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
}

ValueType = Union[bool, int, float, list[float], str]


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
    # Split the last item on the stack into individual items. So if the
    # stack looked like [x,y,[z,w,v]] it would become [x,y,z,w,v].
    SPLIT_STRUCT = auto()
    # Call the given function, all the arguments are on the stack. The value
    # on top of the stack is a list of the inputs for which arguments are
    # provided. Push the output onto the stack.
    CALL_FUNCTION = auto()
    # Same as CALL_FUNCTION but a node group is created.
    CALL_NODEGROUP = auto()
    # Create the built-in node. Data is a NodeInstance with the key set to
    # bl_name.
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
    # A list of the properties that can be set on the node along with their possible values.
    props: list[tuple[str, list[str]]]


@dataclass
class NodeInstance:
    # Key in the nodes dict
    key: str
    # Input indices that are used
    inputs: list[int]
    # Output indices that are used
    outputs: list[int]
    # Props that are set along with their value
    props: list[tuple[str, str]]


class Operation():
    def __init__(self, op_type: OpType, data) -> None:
        self.op_type = op_type
        self.data = data

    def __str__(self) -> str:
        return f"({self.op_type.name}, {self.data})"

    def __repr__(self) -> str:
        return self.__str__()


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
