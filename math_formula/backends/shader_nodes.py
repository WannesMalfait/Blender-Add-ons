from math_formula.backends.builtin_nodes import nodes, instances, levenshtein_distance
from math_formula.backends.main import BackEnd
from math_formula.backends.type_defs import *

shader_nodes = {
    'tex_coords': [NodeInstance('ShaderNodeTexCoord', [], [0, 1, 2, 3, 4, 5, 6], [])],
    'normal': [NodeInstance('ShaderNodeTexCoord', [], [1], [])],
    'geometry': [NodeInstance('ShaderNodeNewGeometry', [], [0, 1, 2, 3, 4, 5, 6, 7], [])],
    'position': [NodeInstance('ShaderNodeNewGeometry', [], [0], [])],
}


class ShaderNodesBackEnd(BackEnd):

    def create_input(self, operations: list[Operation], name: str, value: ValueType, dtype: DataType):
        return super().create_input(operations, name, value, dtype, input_vector=False)

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        if type.value > DataType.VEC3:
            raise TypeError(
                f'Can\'t coerce type {type._name_} to a Shader Nodes value')
        if type == DataType.INT or type == DataType.BOOL:
            value = self.convert(value, type, DataType.FLOAT)
            type = DataType.FLOAT
        return value, type

    @staticmethod
    def input_types(instance: NodeInstance) -> list[DataType]:
        node = nodes[instance.key]
        return [node.inputs[i][1] for i in instance.inputs]

    def resolve_function(self, name: str, args: list[ty_expr]) -> tuple[NodeInstance, list[DataType], list[str]]:
        instance_options: list[NodeInstance] = []
        if name in shader_nodes:
            instance_options += shader_nodes[name]
        if name in instances:
            instance_options += instances[name]
        if instance_options == []:
            # Try to get a suggestion in case of a typo.
            options = sorted(list(shader_nodes.keys()) + list(instances.keys()),
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
