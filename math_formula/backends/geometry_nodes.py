from math_formula.backends.builtin_nodes import nodes, instances, levenshtein_distance
from math_formula.backends.main import BackEnd
from math_formula.backends.type_defs import *

geometry_nodes = {
    'and': [NodeInstance('FunctionNodeBooleanMath', [0, 1], [0], [('operation', 'AND')])],
    'or': [NodeInstance('FunctionNodeBooleanMath', [0, 1], [0], [('operation', 'OR')])],
    'not': [NodeInstance('FunctionNodeBooleanMath', [0], [0], [('operation', 'NOT')])],
    'less_than': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'LESS_THAN')]),
                  NodeInstance('FunctionNodeCompare', [2, 3], [0],
                               [('operation', 'LESS_THAN'), ('data_type', 'INT')]),
                  NodeInstance('FunctionNodeCompare', [4, 5], [0],
                               [('operation', 'LESS_THAN'), ('data_type', 'VECTOR')]),
                  NodeInstance('FunctionNodeCompare', [6, 7], [0], [('operation', 'DARKER'), ('data_type', 'RGBA')])],
    'less_equal': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'LESS_EQUAL')]),
                   NodeInstance('FunctionNodeCompare', [2, 3], [0],
                                [('operation', 'LESS_EQUAL'), ('data_type', 'INT')]),
                   NodeInstance('FunctionNodeCompare', [4, 5], [0], [('operation', 'LESS_EQUAL'), ('data_type', 'VECTOR')])],
    'greater_than': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'GREATER_THAN')]),
                     NodeInstance('FunctionNodeCompare', [2, 3], [
                                  0], [('operation', 'GREATER_THAN'), ('data_type', 'INT')]),
                     NodeInstance('FunctionNodeCompare', [4, 5], [0], [
                                  ('operation', 'GREATER_THAN'), ('data_type', 'VECTOR')]),
                     NodeInstance('FunctionNodeCompare', [6, 7], [0], [('operation', 'LIGHTER'), ('data_type', 'RGBA')])],
    'greater_equal': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'GREATER_EQUAL')]),
                      NodeInstance('FunctionNodeCompare', [2, 3], [
                                   0], [('operation', 'GREATER_EQUAL'), ('data_type', 'INT')]),
                      NodeInstance('FunctionNodeCompare', [4, 5], [0], [
                                   ('operation', 'GREATER_EQUAL'), ('data_type', 'VECTOR')])],
    'equal': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'EQUAL')]),
              NodeInstance('FunctionNodeCompare', [2, 3], [
                  0], [('operation', 'EQUAL'), ('data_type', 'INT')]),
              NodeInstance('FunctionNodeCompare', [4, 5], [0], [
                  ('operation', 'EQUAL'), ('data_type', 'VECTOR')]),
              NodeInstance('FunctionNodeCompare', [6, 7], [0], [
                           ('operation', 'EQUAL'), ('data_type', 'RGBA')]),
              NodeInstance('FunctionNodeCompare', [8, 9], [0], [('operation', 'EQUAL'), ('data_type', 'RGBA')])],
    'not_equal': [NodeInstance('FunctionNodeCompare', [0, 1], [0], [('operation', 'NOT_EQUAL')]),
                  NodeInstance('FunctionNodeCompare', [2, 3], [
                      0], [('operation', 'NOT_EQUAL'), ('data_type', 'INT')]),
                  NodeInstance('FunctionNodeCompare', [4, 5], [0], [
                      ('operation', 'NOT_EQUAL'), ('data_type', 'VECTOR')]),
                  NodeInstance('FunctionNodeCompare', [6, 7], [0], [
                      ('operation', 'NOT_EQUAL'), ('data_type', 'RGBA')]),
                  NodeInstance('FunctionNodeCompare', [8, 9], [0], [('operation', 'NOT_EQUAL'), ('data_type', 'RGBA')])],
    'rand': [NodeInstance('FunctionNodeRandomValue', [0, 1, 7, 8], [0], [('data_type', 'FLOAT_VECTOR')]),
             NodeInstance('FunctionNodeRandomValue', [2, 3, 7, 8], [1], [('data_type', 'FLOAT')])],
    'rand_int': [NodeInstance('FunctionNodeRandomValue', [4, 5, 7, 8], [2], [('data_type', 'INT')])],
    'rand_bool': [NodeInstance('FunctionNodeRandomValue', [6, 7, 8], [3], [('data_type', 'BOOLEAN')])],
    'is_viewport': [NodeInstance('GeometryNodeIsViewport', [], [0], [])],
    'id': [NodeInstance('GeometryNodeInputID', [], [0], [])],
    'position': [NodeInstance('GeometryNodeInputPosition', [], [0], [])],
    'normal': [NodeInstance('GeometryNodeInputNormal', [], [0], [])],
    'index': [NodeInstance('GeometryNodeInputIndex', [], [0], [])],
    'time': [NodeInstance('GeometryNodeInputSceneTime', [], [0, 1], [])],
    'radius': [NodeInstance('GeometryNodeInputRadius', [], [0], [])],
}


class GeometryNodesBackEnd(BackEnd):

    def create_input(self, operations: list[Operation], name: str, value: ValueType, dtype: DataType):
        return super().create_input(operations, name, value, dtype, input_vector=True)

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        if type.value >= DataType.SHADER or type.value == DataType.GEOMETRY:
            raise TypeError(
                f'Can\'t coerce type {type._name_} to a Geometry Nodes value')
        return value, type

    @staticmethod
    def input_types(instance: NodeInstance) -> list[DataType]:
        node = nodes[instance.key]
        return [node.inputs[i][1] for i in instance.inputs]

    def resolve_function(self, name: str, args: list[ty_expr]) -> tuple[NodeInstance, list[DataType], list[str]]:
        instance_options: list[NodeInstance] = []
        if name in geometry_nodes:
            instance_options += geometry_nodes[name]
        if name in instances:
            instance_options += instances[name]
        if instance_options == []:
            # Try to get a suggestion in case of a typo.
            options = sorted(list(geometry_nodes.keys()) + list(instances.keys()),
                             key=lambda x: levenshtein_distance(name, x))
            raise TypeError(
                f'No function named "{name}" found. Did you mean "{options[0]}" or "{options[1]}"?')
        options = [self.input_types(option) for option in instance_options]
        index = self.find_best_match(options, args, name)
        func = instance_options[index]
        node = nodes[func.key]
        out_types = [node.outputs[i][1] for i in func.outputs]
        out_names = [node.outputs[i][0] for i in func.outputs]
        return func, out_types, out_names
