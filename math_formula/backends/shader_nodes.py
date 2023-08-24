from . import type_defs as td
from .builtin_nodes import instances, shader_geo_node_aliases, shader_node_aliases
from .main import BackEnd

shader_nodes: dict[str, list[str | td.NodeInstance]] = {
    "tex_coords": ["texture_coordinate"],
    "normal": [td.NodeInstance("ShaderNodeTexCoord", [], [1], [])],
    "geometry": [
        td.NodeInstance("ShaderNodeNewGeometry", [], [0, 1, 2, 3, 4, 5, 6, 7], [])
    ],
    "position": [td.NodeInstance("ShaderNodeNewGeometry", [], [0], [])],
}


class ShaderNodesBackEnd(BackEnd):
    def create_input(
        self,
        operations: list[td.Operation],
        name: str,
        value: td.ValueType,
        dtype: td.DataType,
    ):
        return super().create_input_helper(
            operations, name, value, dtype, input_vector=False
        )

    def coerce_value(
        self, value: td.ValueType, type: td.DataType
    ) -> tuple[td.ValueType, td.DataType]:
        if type.value > td.DataType.VEC3:
            raise TypeError(f"Can't coerce type {type._name_} to a Shader Nodes value")
        if type == td.DataType.INT or type == td.DataType.BOOL:
            value = self.convert(value, type, td.DataType.FLOAT)
            type = td.DataType.FLOAT
        return value, type

    def resolve_function(
        self,
        name: str,
        args: list[td.ty_expr],
        functions: dict[str, list[td.TyFunction]],
    ) -> tuple[td.TyFunction | td.NodeInstance, list[td.DataType], list[str]]:
        return self._resolve_function(
            name,
            args,
            [shader_node_aliases, shader_geo_node_aliases],
            [shader_nodes, instances, functions],
        )
