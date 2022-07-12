from math_formula.backends.type_defs import *
from math_formula.backends.builtin_nodes import levenshtein_distance, nodes
from functools import reduce


class BackEnd():

    @staticmethod
    def can_convert(from_type: DataType, to_type: DataType) -> bool:
        if from_type == to_type:
            return True
        else:
            return from_type.value <= DataType.VEC3.value and to_type.value <= DataType.VEC3.value

    def convert(self, value: ValueType, from_type: DataType, to_type: DataType) -> ValueType:
        '''Convert value of type from_type to to_type.'''
        assert self.can_convert(
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

    def create_input(self, operations: list[Operation], name: str, value: ValueType, dtype: DataType, input_vector=True):
        if dtype == DataType.FLOAT or dtype == DataType.UNKNOWN:
            operations.append(
                Operation(OpType.CALL_BUILTIN,
                          NodeInstance('ShaderNodeValue', [], [0], [])))
            if value:
                operations.append(
                    Operation(OpType.SET_OUTPUT, (0, value)))
        elif dtype == DataType.BOOL:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        NodeInstance('FunctionNodeInputBool', [], [0],
                                                     [('boolean', value)] if value is not None else [])))
        elif dtype == DataType.INT:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        NodeInstance('FunctionNodeInputInt', [], [0],
                                                     [('integer', value)] if value is not None else [])))
        elif dtype == DataType.RGBA:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        NodeInstance('FunctionNodeInputColor', [], [0],
                                                     [('color', value)] if value is not None else [])))
        elif dtype == DataType.VEC3:
            # Only geometry nodes has this input vector node
            if input_vector:
                operations.append(Operation(OpType.CALL_BUILTIN,
                                            NodeInstance('FunctionNodeInputVector', [], [0],
                                                         [('vector', value)] if value is not None else [])))
            else:
                if value is not None:
                    for v in value:
                        operations.append(Operation(OpType.PUSH_VALUE, v))
                else:
                    for _ in range(3):
                        operations.append(Operation(OpType.PUSH_VALUE, None))
                operations.append(Operation(OpType.CALL_BUILTIN,
                                            NodeInstance('ShaderNodeCombineXYZ', [0, 1, 2], [0], [])))
        elif dtype == DataType.STRING:
            operations.append(Operation(OpType.CALL_BUILTIN,
                                        NodeInstance('FunctionNodeInputString', [], [0],
                                                     [('string', value)] if value is not None else [])))
        else:
            raise NotImplementedError(f'Creating input of type {str(dtype)}')
        operations.append(Operation(OpType.RENAME_NODE, name))
        operations.append(Operation(OpType.CREATE_VAR, name))

    def find_best_match(self, options: list[list[DataType]], args: list[ty_expr], name: str) -> int:
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
            if len(option) < len(arg_types):
                # If we pass more arguments than the function accepts
                # it can never be a match. If we pass less arguments
                # the rest are implicit default arguments.
                continue
            penalty = sum([dtype_conversion_penalties[arg_types[i].value]
                           [option[i].value] for i in range(len(arg_types))])

            if best_penalty > penalty:
                best_penalty = penalty
                best_index = i
            if best_penalty == 0:
                break
        if best_penalty < 100:
            # Ensure that the arguments are of the correct type
            for arg, otype in zip(args, options[best_index]):
                if isinstance(arg, Const):
                    arg.value = self.convert(arg.value, arg.dtype[0], otype)
                arg.dtype[0] = otype
            return best_index
        # print(f'\nOPTIONS: {options}\nARGS: {arg_types}')
        raise TypeError(
            f'Couldn\'t find find instance of function "{name}" with arguments {arg_types}')

    @staticmethod
    def input_types(instance: Union[NodeInstance, TyFunction]) -> list[DataType]:
        if isinstance(instance, NodeInstance):
            node = nodes[instance.key]
            return [node.inputs[i][1] for i in instance.inputs]
        else:
            return [i.dtype for i in instance.inputs]

    def _resolve_function(self, name: str, args: list[DataType], dicts: list[dict]) -> tuple[Union[TyFunction, NodeInstance], list[DataType], list[str]]:
        instance_options: list[Union[NodeInstance, TyFunction]] = []
        for dict in dicts:
            if name in dict:
                instance_options += dict[name]
        if instance_options == []:
            # Try to get a suggestion in case of a typo.
            options = sorted(reduce(lambda a, b: a + b, map(lambda x: list(x.keys()), dicts)),
                             key=lambda x: levenshtein_distance(name, x))
            raise TypeError(
                f'No function named "{name}" found. Did you mean "{options[0]}" or "{options[1]}"?')
        options = [self.input_types(option) for option in instance_options]
        index = self.find_best_match(options, args, name)
        func = instance_options[index]
        if isinstance(func, TyFunction):
            out_types = [o.dtype for o in func.outputs]
            out_names = [o.name for o in func.outputs]
            return func, out_types, out_names
        node = nodes[func.key]
        out_types = [node.outputs[i][1] for i in func.outputs]
        out_names = [node.outputs[i][0] for i in func.outputs]
        return func, out_types, out_names

    def resolve_function(self, name: str, args: list[DataType], functions: list[TyFunction]) -> tuple[Union[TyFunction, NodeInstance], list[DataType], list[str]]:
        ''' Resolve name to a built-in node by type matching on the arguments.'''
        pass
