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


def can_convert(from_type: DataType, to_type: DataType):
    if from_type == to_type:
        return True
    else:
        return from_type.value <= DataType.VEC3.value and to_type.value <= DataType.VEC3.value


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
    # Set the ouput of the last added node to the given value. Data is a
    # tuple of the output index and the value to be set.
    SET_OUTPUT = auto()
    # Call the given function, all the arguments are on the stack. The value
    # on top of the stack is a list of the inputs for which arguments are
    # provided. Push the output onto the stack.
    CALL_FUNCTION = auto()
    # Same as CALL_FUNCTION but a node group is created.
    CALL_NODEGROUP = auto()
    # Create the built-in node. Data is a tuple containing the bl_name and
    # props to be set. The value at the top of the stack is a tuple containing
    # a list of inputs and a list of outputs of the node that are used.
    CALL_BUILTIN = auto()
    # Set the label of the last added node to the given name.
    RENAME_NODE = auto()
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

    def convert(self, value: ValueType, from_type: DataType, to_type: DataType) -> ValueType:
        '''Convert value of type from_type to to_type.'''
        assert can_convert(
            from_type, to_type), f'Invalid type, can\'t convert from {from_type} to {to_type} '
        if from_type == DataType.DEFAULT:
            if to_type == DataType.BOOL:
                return True
            if to_type == DataType.INT:
                return 0
            if to_type == DataType.FLOAT:
                return 0.0
            if to_type == DataType.VEC3:
                return [0.0, 0.0, 0.0]
            if to_type == DataType.RGBA:
                return [0.0, 0.0, 0.0, 0.0]
        if from_type == DataType.BOOL:
            if to_type == DataType.INT:
                return int(value)
            if to_type == DataType.FLOAT:
                return float(value)
            if to_type == DataType.RGBA:
                return [float(value) for _ in range(4)]
            if to_type == DataType.VEC3:
                return [float(value) for _ in range(3)]
        if from_type == DataType.INT:
            if to_type == DataType.BOOL:
                return bool(value <= 0)
            if to_type == DataType.FLOAT:
                return float(value)
            if to_type == DataType.RGBA:
                return [float(value) for _ in range(4)]
            if to_type == DataType.VEC3:
                return [float(value) for _ in range(3)]
        if from_type == DataType.FLOAT:
            if to_type == DataType.BOOL:
                return bool(value <= 0.0)
            if to_type == DataType.INT:
                return int(value)
            if to_type == DataType.RGBA:
                return [value for _ in range(4)]
            if to_type == DataType.VEC3:
                return [value for _ in range(3)]
        if from_type == DataType.RGBA:
            gray_scale = (
                0.2126 * value[0]) + (0.7152 * value[1]) + (0.0722 * value[2])
            if to_type == DataType.BOOL:
                return bool(gray_scale)
            if to_type == DataType.INT:
                return int(gray_scale)
            if to_type == DataType.FLOAT:
                return gray_scale
            if to_type == DataType.VEC3:
                return [value[i] for i in range(3)]
        if from_type == DataType.VEC3:
            avg = (
                value[0] + value[1] + value[2])/3.0
            if to_type == DataType.BOOL:
                return bool(avg)
            if to_type == DataType.INT:
                return int(avg)
            if to_type == DataType.FLOAT:
                return avg
            if to_type == DataType.RGBA:
                return value + [1]
        return value

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        '''Ensure that the value is of a type supported by the backend'''
        pass

    def create_input(self, operations: list[Operation], name: str, value: ValueType, dtype: DataType) -> DataType:
        value, dtype = self.coerce_value(value, dtype)
        if dtype == DataType.DEFAULT or dtype == DataType.UNKNOWN:
            dtype = DataType.FLOAT
            value = 0.0
        # No inputs and one output to be used
        operations.append(
            Operation(OpType.PUSH_VALUE, ([], [0])))

        if dtype == DataType.FLOAT:
            operations.append(
                Operation(OpType.CALL_BUILTIN, ('ShaderNodeValue', [])))
            operations.append(
                Operation(OpType.SET_OUTPUT, (0, value)))
        elif dtype == DataType.BOOL:
            Operation(OpType.CALL_BUILTIN,
                      ('FunctionNodeInputBool', [('boolean', value)]))
        elif dtype == DataType.INT:
            Operation(OpType.CALL_BUILTIN,
                      ('FunctionNodeInputInt', [('integer', value)]))
        elif dtype == DataType.RGBA:
            Operation(OpType.CALL_BUILTIN,
                      ('FunctionNodeInputColor', [('color', value)]))
        elif dtype == DataType.STRING:
            Operation(OpType.CALL_BUILTIN,
                      ('FunctionNodeInputString', [('string', value)]))
        else:
            raise NotImplementedError(f'Creating input of type {dtype}')
        operations.append(Operation(OpType.RENAME_NODE, name))
        operations.append(Operation(OpType.CREATE_VAR, name))
        return dtype

    def find_best_match(self, options: list[list[DataType]], args: list[DataType]) -> int:
        '''Find the best function to use from the list of options.
        The options argument contains a list of possible function argument types.
        Returns the index of the best match.'''

        # Find the one with the least amount of penalty
        # penalty = 0 means a perfect match
        best_penalty = 100
        best_index = 0
        for i, option in enumerate(options):
            if len(option) != len(args):
                continue
            penalty = sum([dtype_conversion_penalties[args[i].value]
                           [option[i].value] for i in range(len(option))])
            if penalty == 0:
                return i
            if best_penalty > penalty:
                best_penalty = penalty
                best_index = i
        if best_penalty < 100:
            return best_index
        print(f'\nOPTIONS: {options}\nARGS: {args}')
        assert False, "No matching option found"

    def compile_function(self, operations: list[Operation], name: str, args: list[DataType], stack_locs: list[int]) -> DataType:
        ''' Compile function with given arguments and return the return type of the function'''
        pass
