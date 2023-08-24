from abc import ABCMeta, abstractmethod
from typing import Literal, overload

from . import type_defs as td
from .builtin_nodes import levenshtein_distance, nodes


class BackEnd(ABCMeta):

    @staticmethod
    def can_convert(from_type: td.DataType, to_type: td.DataType) -> bool:
        if from_type == to_type or from_type == td.DataType.DEFAULT:
            return True
        else:
            return from_type.value <= td.DataType.VEC3.value and to_type.value <= td.DataType.VEC3.value

    @overload
    def convert(self, value: td.ValueType,
                from_type: td.DataType, to_type: Literal[td.DataType.BOOL]) -> bool: ...

    @overload
    def convert(self, value: td.ValueType,
                from_type: td.DataType, to_type: Literal[td.DataType.INT]) -> int: ...

    @overload
    def convert(self, value: td.ValueType,
                from_type: td.DataType, to_type: Literal[td.DataType.FLOAT]) -> float: ...

    @overload
    def convert(self, value: td.ValueType,
                from_type: td.DataType, to_type: Literal[td.DataType.VEC3]) -> list[float]: ...

    @overload
    def convert(self, value: td.ValueType,
                from_type: td.DataType, to_type: Literal[td.DataType.RGBA]) -> list[float]: ...

    @overload
    def convert(self, value: td.ValueType, from_type: td.DataType,
                to_type: td.DataType) -> td.ValueType: ...

    def convert(self, value: td.ValueType, from_type: td.DataType, to_type: td.DataType) -> td.ValueType:
        '''Convert value of type from_type to to_type.'''
        assert self.can_convert(
            from_type, to_type), f'Invalid type, can\'t convert from {from_type} to {to_type} '
        if from_type == td.DataType.DEFAULT or from_type == td.DataType.UNKNOWN:
            if to_type == td.DataType.BOOL:
                return True
            if to_type == td.DataType.INT:
                return 0
            if to_type == td.DataType.FLOAT:
                return 0.0
            if to_type == td.DataType.VEC3:
                return [0.0, 0.0, 0.0]
            if to_type == td.DataType.RGBA:
                return [0.0, 0.0, 0.0, 0.0]
        if from_type == td.DataType.BOOL:
            assert isinstance(value, bool)
            if to_type == td.DataType.INT:
                return int(value)
            if to_type == td.DataType.FLOAT:
                return float(value)
            if to_type == td.DataType.RGBA:
                return [float(value) for _ in range(4)]
            if to_type == td.DataType.VEC3:
                return [float(value) for _ in range(3)]
        if from_type == td.DataType.INT:
            assert isinstance(value, int)
            if to_type == td.DataType.BOOL:
                return bool(value <= 0)
            if to_type == td.DataType.FLOAT:
                return float(value)
            if to_type == td.DataType.RGBA:
                return [float(value) for _ in range(4)]
            if to_type == td.DataType.VEC3:
                return [float(value) for _ in range(3)]
        if from_type == td.DataType.FLOAT:
            assert isinstance(value, float)
            if to_type == td.DataType.BOOL:
                return bool(value <= 0.0)
            if to_type == td.DataType.INT:
                return int(value)
            if to_type == td.DataType.RGBA:
                return [value for _ in range(4)]
            if to_type == td.DataType.VEC3:
                return [value for _ in range(3)]
        if from_type == td.DataType.RGBA:
            assert isinstance(value, list)
            gray_scale = (
                0.2126 * value[0]) + (0.7152 * value[1]) + (0.0722 * value[2])
            if to_type == td.DataType.BOOL:
                return bool(gray_scale)
            if to_type == td.DataType.INT:
                return int(gray_scale)
            if to_type == td.DataType.FLOAT:
                return gray_scale
            if to_type == td.DataType.VEC3:
                return [value[i] for i in range(3)]
        if from_type == td.DataType.VEC3:
            assert isinstance(value, list)
            avg = (
                value[0] + value[1] + value[2])/3.0
            if to_type == td.DataType.BOOL:
                return bool(avg)
            if to_type == td.DataType.INT:
                return int(avg)
            if to_type == td.DataType.FLOAT:
                return avg
            if to_type == td.DataType.RGBA:
                return value + [1]
        return value

    @abstractmethod
    def coerce_value(self, value: td.ValueType, type: td.DataType) -> tuple[td.ValueType, td.DataType]:
        '''Ensure that the value is of a type supported by the backend'''
        ...

    def create_input(
            self,
            operations: list[td.Operation],
            name: str,
            value: td.ValueType | None,
            dtype: td.DataType,
            input_vector=True):
        if dtype == td.DataType.FLOAT or dtype == td.DataType.UNKNOWN:
            operations.append(
                td.Operation(td.OpType.CALL_BUILTIN,
                             td.NodeInstance('ShaderNodeValue', [], [0], [])))
            if value:
                operations.append(
                    td.Operation(td.OpType.SET_OUTPUT, (0, value)))
        elif dtype == td.DataType.BOOL:
            operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                           td.NodeInstance('FunctionNodeInputBool', [], [0],
                                                           [('boolean', value)] if value is not None else [])))
        elif dtype == td.DataType.INT:
            operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                           td.NodeInstance('FunctionNodeInputInt', [], [0],
                                                           [('integer', value)] if value is not None else [])))
        elif dtype == td.DataType.RGBA:
            operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                           td.NodeInstance('FunctionNodeInputColor', [], [0],
                                                           [('color', value)] if value is not None else [])))
        elif dtype == td.DataType.VEC3:
            # Only geometry nodes has this input vector node
            if input_vector:
                operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                               td.NodeInstance('FunctionNodeInputVector', [], [0],
                                                               [('vector', value)] if value is not None else [])))
            else:
                if value is not None:
                    assert isinstance(
                        value, list), 'Vec3 should be list of floats'
                    for v in value:
                        operations.append(td.Operation(
                            td.OpType.PUSH_VALUE, v))
                else:
                    for _ in range(3):
                        operations.append(td.Operation(
                            td.OpType.PUSH_VALUE, None))
                operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                               td.NodeInstance('ShaderNodeCombineXYZ', [0, 1, 2], [0], [])))
        elif dtype == td.DataType.STRING:
            operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                           td.NodeInstance('FunctionNodeInputString', [], [0],
                                                           [('string', value)] if value is not None else [])))
        else:
            raise NotImplementedError(f'Creating input of type {str(dtype)}')
        operations.append(td.Operation(td.OpType.RENAME_NODE, name))

    def find_best_match(self, options: list[list[td.DataType]], args: list[td.ty_expr], name: str) -> int:
        '''Find the best function to use from the list of options.
        The options argument contains a list of possible function argument types.
        Returns the index of the best match.'''

        # NOTE: This assumes that no 'empty' arguments were passed.
        #       Those should have been handled before this.
        arg_types = [arg.dtype[0] for arg in args]

        # Find the one with the least amount of penalty
        # penalty = 0 means a perfect match
        # penalty >= 1000 means no match, or match that is extremely bad
        best_penalty = 1000
        best_index = 0
        for i, option in enumerate(options):
            if len(option) < len(arg_types):
                # If we pass more arguments than the function accepts
                # it can never be a match. If we pass less arguments
                # the rest are implicit default arguments.
                continue
            penalty = sum([td.dtype_conversion_penalties[arg_types[i].value]
                           [option[i].value] for i in range(len(arg_types))])

            if best_penalty > penalty:
                best_penalty = penalty
                best_index = i
            if best_penalty == 0:
                break
        if best_penalty < 1000:
            # Ensure that the arguments are of the correct type
            for arg, otype in zip(args, options[best_index]):
                if isinstance(arg, td.Const):
                    arg.value = self.convert(arg.value, arg.dtype[0], otype)
                arg.dtype[0] = otype
            return best_index
        # print(f'\nOPTIONS: {options}\nARGS: {arg_types}')
        raise TypeError(
            f'Couldn\'t find find instance of function "{name}" with arguments {arg_types}')

    @staticmethod
    def input_types(instance: td.Union[td.NodeInstance, td.TyFunction]) -> list[td.DataType]:
        if isinstance(instance, td.NodeInstance):
            node = nodes[instance.key]
            return [node.inputs[i][1] for i in instance.inputs]
        else:
            return [i.dtype for i in instance.inputs]

    @staticmethod
    def resolve_alias(
        thing: td.NodeInstance | td.TyFunction | str,
        aliases: list[dict[str, td.NodeInstance]]
    ) -> td.NodeInstance | td.TyFunction:
        if not isinstance(thing, str):
            return thing
        for alias_dict in aliases:
            if thing in alias_dict:
                return alias_dict[thing]

        assert False, 'Unreachable, aliases should be valid!'

    def _resolve_function(
        self,
        name: str,
        args: list[td.ty_expr],
        aliases: list[dict[str, td.NodeInstance]],
        dicts: list[dict[str, list[str | td.NodeInstance]]
                    | dict[str, list[td.TyFunction]]]
    ) -> tuple[td.Union[td.TyFunction, td.NodeInstance], list[td.DataType], list[str]]:
        instance_options: list[td.NodeInstance | td.TyFunction] = []
        for dict in dicts:
            if name in dict:
                options = dict[name]
                resolved_options = list(
                    map(lambda thing: self.resolve_alias(thing, aliases), options))
                instance_options += resolved_options
        if instance_options == []:
            # Only check these if the other thing failed.
            for alias_dict in aliases:
                if name in alias_dict:
                    instance_options.append(alias_dict[name])
        if instance_options == []:
            # Try to get a suggestion in case of a typo.
            keys = []
            for d in dicts:
                keys += list(d.keys())
            for a in aliases:
                keys += list(a.keys())
            suggestions = sorted(keys,
                                 key=lambda x: levenshtein_distance(name, x))
            raise TypeError(
                f'No function named "{name}" found. Did you mean "{suggestions[0]}" or "{suggestions[1]}"?')
        all_options = [self.input_types(option) for option in instance_options]
        index = self.find_best_match(all_options, args, name)
        func = instance_options[index]
        if isinstance(func, td.TyFunction):
            out_types = [o.dtype for o in func.outputs]
            out_names = [o.name for o in func.outputs]
            return func, out_types, out_names
        node = nodes[func.key]
        out_types = [node.outputs[i][1] for i in func.outputs]
        out_names = [node.outputs[i][0] for i in func.outputs]
        return func, out_types, out_names

    @ abstractmethod
    def resolve_function(
        self,
        name: str,
        args: list[td.ty_expr],
        functions: dict[str, list[td.TyFunction]]
    ) -> tuple[td.Union[td.TyFunction, td.NodeInstance], list[td.DataType], list[str]]:
        ''' Resolve name to a built-in node by type matching on the arguments.'''
        ...
