from math_formula.backends.builtin_nodes import nodes, instances
from math_formula.backends.main import BackEnd
from math_formula.backends.type_defs import *

geometry_nodes = {
    'and': [NodeInstance('Boolean Math', [0, 1], [0], [('operation', 'AND')])],
    'or': [NodeInstance('Boolean Math', [0, 1], [0], [('operation', 'OR')])],
    'not': [NodeInstance('Boolean Math', [0], [0], [('operation', 'NOT')])],
    'less_than': [NodeInstance('Compare', [0, 1], [0], [('operation', 'LESS_THAN')]),
                  NodeInstance('Compare', [2, 3], [0],
                               [('operation', 'LESS_THAN'), ('data_type', 'INT')]),
                  NodeInstance('Compare', [4, 5], [0],
                               [('operation', 'LESS_THAN'), ('data_type', 'VECTOR')]),
                  NodeInstance('Compare', [6, 7], [0], [('operation', 'DARKER'), ('data_type', 'RGBA')])],
    'less_equal': [NodeInstance('Compare', [0, 1], [0], [('operation', 'LESS_EQUAL')]),
                   NodeInstance('Compare', [2, 3], [0],
                                [('operation', 'LESS_EQUAL'), ('data_type', 'INT')]),
                   NodeInstance('Compare', [4, 5], [0], [('operation', 'LESS_EQUAL'), ('data_type', 'VECTOR')])],
    'greater_than': [NodeInstance('Compare', [0, 1], [0], [('operation', 'GREATER_THAN')]),
                     NodeInstance('Compare', [2, 3], [
                                  0], [('operation', 'GREATER_THAN'), ('data_type', 'INT')]),
                     NodeInstance('Compare', [4, 5], [0], [
                                  ('operation', 'GREATER_THAN'), ('data_type', 'VECTOR')]),
                     NodeInstance('Compare', [6, 7], [0], [('operation', 'LIGHTER'), ('data_type', 'RGBA')])],
    'greater_equal': [NodeInstance('Compare', [0, 1], [0], [('operation', 'GREATER_EQUAL')]),
                      NodeInstance('Compare', [2, 3], [
                                   0], [('operation', 'GREATER_EQUAL'), ('data_type', 'INT')]),
                      NodeInstance('Compare', [4, 5], [0], [
                                   ('operation', 'GREATER_EQUAL'), ('data_type', 'VECTOR')])],
    'equal': [NodeInstance('Compare', [0, 1], [0], [('operation', 'EQUAL')]),
              NodeInstance('Compare', [2, 3], [
                  0], [('operation', 'EQUAL'), ('data_type', 'INT')]),
              NodeInstance('Compare', [4, 5], [0], [
                  ('operation', 'EQUAL'), ('data_type', 'VECTOR')]),
              NodeInstance('Compare', [6, 7], [0], [
                           ('operation', 'EQUAL'), ('data_type', 'RGBA')]),
              NodeInstance('Compare', [8, 9], [0], [('operation', 'EQUAL'), ('data_type', 'RGBA')])],
    'not_equal': [NodeInstance('Compare', [0, 1], [0], [('operation', 'NOT_EQUAL')]),
                  NodeInstance('Compare', [2, 3], [
                      0], [('operation', 'NOT_EQUAL'), ('data_type', 'INT')]),
                  NodeInstance('Compare', [4, 5], [0], [
                      ('operation', 'NOT_EQUAL'), ('data_type', 'VECTOR')]),
                  NodeInstance('Compare', [6, 7], [0], [
                      ('operation', 'NOT_EQUAL'), ('data_type', 'RGBA')]),
                  NodeInstance('Compare', [8, 9], [0], [('operation', 'NOT_EQUAL'), ('data_type', 'RGBA')])],
    'rand': [NodeInstance('Random Value', [0, 1, 7, 8], [0], [('data_type', 'FLOAT_VECTOR')]),
             NodeInstance('Random Value', [2, 3, 7, 8], [1], [('data_type', 'FLOAT')])],
    'rand_int': [NodeInstance('Random Value', [4, 5, 7, 8], [2], [('data_type', 'INT')])],
    'rand_bool': [NodeInstance('Random Value', [6, 7, 8], [3], [('data_type', 'BOOLEAN')])],
    'is_viewport': [NodeInstance('Is Viewport', [], [0], [])],
    'id': [NodeInstance('ID', [], [0], [])],
    'position': [NodeInstance('Position', [], [0], [])],
    'normal': [NodeInstance('Normal', [], [0], [])],
    'index': [NodeInstance('Index', [], [0], [])],
    'time': [NodeInstance('Scene Time', [], [0, 1], [])],
    'radius': [NodeInstance('Radius', [], [0], [])],
}


class GeometryNodesBackEnd(BackEnd):

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        if type.value >= DataType.SHADER or type.value == DataType.GEOMETRY:
            raise TypeError('Can\'t coerce type to a Geometry Nodes value')
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
        options = [self.input_types(option) for option in instance_options]
        index = self.find_best_match(options, args, name)
        func = instance_options[index]
        node = nodes[func.key]
        out_types = [node.outputs[i][1] for i in func.outputs]
        out_names = [node.outputs[i][0] for i in func.outputs]
        return func, out_types, out_names
