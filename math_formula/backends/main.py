from enum import IntEnum, auto
from dataclasses import dataclass
from typing import Any


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


ValueType = Any


class OpType(IntEnum):
    # Push the given value on the stack. None represents a default value.
    PUSH_VALUE = 0
    # Create a variable with the given name, and assign it to stack.pop().
    CREATE_VAR = auto()
    # Get the variable with the given name, and push it onto the stack.
    GET_VAR = auto()
    # Get the output with the given index from the last value on the stack.
    # Put this value on top of the stack.
    GET_OUTPUT = auto()
    # Call the given function, all the arguments are on the stack. The value
    # on top of the stack is a list of the inputs for which arguments are
    # provided. Push the output onto the stack.
    CALL_FUNCTION = auto()
    # Same as CALL_FUNCTION except the function is just a built-in node, and
    # the data is a tuple containing the bl_name and the properties that should
    # be set.
    CALL_BUILTIN = auto()
    # Same as CALL_FUNCTION but a node group is created.
    CALL_NODEGROUP = auto()
    # Create an input node for the given type. The value is stack.pop().
    CREATE_INPUT = auto()
    # Clear the stack.
    END_OF_STATEMENT = auto()


@dataclass
class BuiltinNode:
    # The name used by blender, e.g. 'ShaderNodeMapRange'
    bl_name: str
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


class BackEnd():

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        '''Ensure that the value is of a type supported by the backend'''
        pass

    def find_best_match(self, options: list[list[DataType]], args: list[DataType]) -> int:
        '''Find the best function to use from the list of options.
        The options argument contains a list of possible function argument types.
        Returns the index of the best match.'''

        # For now, assume there is some perfect match:
        for i, option in enumerate(options):
            if option == args:
                return i

        print(f'\nOPTIONS: {options}\nARGS: {args}')
        assert False, "No matching option found"

    def compile_function(self, operations: list[Operation], name: str, args: list[DataType]) -> DataType:
        ''' Compile function with given arguments and return the return type of the function'''
        pass
