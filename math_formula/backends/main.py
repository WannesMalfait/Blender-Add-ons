from math_formula.backends.type_defs import *


def can_convert(from_type: DataType, to_type: DataType):
    if from_type == to_type:
        return True
    else:
        return from_type.value <= DataType.VEC3.value and to_type.value <= DataType.VEC3.value


class BackEnd():

    def convert(self, value: ValueType, from_type: DataType, to_type: DataType) -> ValueType:
        '''Convert value of type from_type to to_type.'''
        assert can_convert(
            from_type, to_type), f'Invalid type, can\'t convert from {from_type} to {to_type} '
        if from_type == DataType.DEFAULT or from_type == DataType.UNKNOWN:
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
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        ('FunctionNodeInputBool', [('boolean', value)])))
        elif dtype == DataType.INT:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        ('FunctionNodeInputInt', [('integer', value)])))
        elif dtype == DataType.RGBA:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        ('FunctionNodeInputColor', [('color', value)])))
        elif dtype == DataType.VEC3:
            # TODO: This doesn't work for shaders
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        ('FunctionNodeInputVector', [('vector', value)])))
        elif dtype == DataType.STRING:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        ('FunctionNodeInputString', [('string', value)])))
        else:
            raise NotImplementedError(f'Creating input of type {dtype}')
        operations.append(Operation(OpType.RENAME_NODE, name))
        operations.append(Operation(OpType.CREATE_VAR, name))
        return dtype

    def find_best_match(self, options: list[list[DataType]], args: list[ty_expr]) -> int:
        '''Find the best function to use from the list of options.
        The options argument contains a list of possible function argument types.
        Returns the index of the best match.'''

        # NOTE: This assumes that no 'empty' arguments were passed.
        #       Those should have been handled before this.
        arg_types = [arg.dtype[0] for arg in args]

        # Find the one with the least amount of penalty
        # penalty = 0 means a perfect match
        # penalty >= 100 means no match, or match that is extremely bad
        best_penalty = 100
        best_index = 0
        for i, option in enumerate(options):
            if len(option) != len(args):
                continue
            penalty = sum([dtype_conversion_penalties[arg_types[i].value]
                           [option[i].value] for i in range(len(option))])
            if penalty == 0:
                return i
            if best_penalty > penalty:
                best_penalty = penalty
                best_index = i
        if best_penalty < 100:
            # Ensure that the arguments are of the correct type
            for arg, otype in zip(args, options[best_index]):
                if isinstance(arg, Const):
                    arg.value = self.convert(arg.value, arg.dtype[0], otype)
                arg.dtype[0] = otype
            return best_index
        print(f'\nOPTIONS: {options}\nARGS: {args}')
        raise TypeError(
            f'Couldn\'t find find instance of function with arguments {arg_types}')

    def resolve_function(self, name: str, args: list[DataType]) -> tuple[NodeInstance, list[DataType]]:
        ''' Resolve name to a built-in node by type matching on the arguments.'''
        pass
