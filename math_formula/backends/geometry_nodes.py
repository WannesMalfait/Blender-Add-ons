from math_formula.backends.builtin_nodes import nodes, instances
from math_formula.backends.main import BackEnd, Operation, OpType, DataType, NodeInstance, ValueType

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
        # All the types are valid in Geometry nodes

        # Temporary hack because overloading type matching isn't implemented yet
        # TODO: remove when type matching works better.
        if type == DataType.INT:
            value = float(value)
            type = DataType.FLOAT
        return value, type

    @staticmethod
    def input_types(instance: NodeInstance) -> list[DataType]:
        node = nodes[instance.key]
        return [node.inputs[i][1] for i in instance.inputs]

    def compile_function(self, operations: list[Operation], name: str, args: list[DataType]) -> DataType:
        instance_options: list[NodeInstance] = []
        if name in geometry_nodes:
            instance_options += geometry_nodes[name]
        if name in instances:
            instance_options += instances[name]
        options = [self.input_types(option) for option in instance_options]
        index = self.find_best_match(options, args)
        func = instance_options[index]
        node = nodes[func.key]
        # Add the indices of the inputs that should be connected to
        # And the outputs of the node that can be used
        operations.append(Operation(OpType.PUSH_VALUE,
                                    (func.inputs, func.outputs)))
        operations.append(Operation(OpType.CALL_BUILTIN,
                                    (node.bl_name, func.props)))
        outs = [node.outputs[i][1] for i in func.outputs]
        if len(outs) == 1:
            return outs[0]
        else:
            return DataType.DEFAULT
