from .builtin_nodes import instances, geometry_node_aliases, shader_geo_node_aliases
from .main import BackEnd
from .type_defs import *

geometry_nodes: dict[str, list[str | NodeInstance]] = {
    '_and': ['boolean_math_and'],
    '_or': ['boolean_math_or'],
    '_not': ['boolean_math_not'],
    'round': ['float_to_integer_round'],
    'floor': ['float_to_integer_floor'],
    'ceil': ['float_to_integer_ceiling'],
    'trunc': ['float_to_integer_truncate'],
    'less_than': ['compare_less_than_float_element_wise',
                  'compare_less_than_integer_element_wise',
                  'compare_less_than_vector_element_wise',
                  'compare_darker_color_element_wise'],
    'less_equal': ['compare_less_equal_float_element_wise',
                   'compare_less_equal_integer_element_wise',
                   'compare_less_equal_vector_element_wise'],
    'greater_than': ['compare_greater_than_float_element_wise',
                     'compare_greater_than_integer_element_wise',
                     'compare_greater_than_vector_element_wise',
                     'compare_brighter_color_element_wise'],
    'greater_equal': ['compare_greater_equal_float_element_wise',
                      'compare_greater_equal_integer_element_wise',
                      'compare_greater_equal_vector_element_wise'],
    'equal': ['compare_equal_float_element_wise',
              'compare_equal_integer_element_wise',
              'compare_equal_vector_element_wise',
              'compare_equal_color_element_wise'],
    'not_equal': ['compare_not_equal_float_element_wise',
                  'compare_not_equal_integer_element_wise',
                  'compare_not_equal_vector_element_wise',
                  'compare_not_equal_color_element_wise'],
    'rand': ['random_value_float', 'random_value_vector'],
    'rand_int': ['random_value_integer'],
    'rand_bool': ['random_value_boolean'],
}


class GeometryNodesBackEnd(BackEnd):

    def create_input(self, operations: list[Operation], name: str, value: ValueType, dtype: DataType):
        return super().create_input(operations, name, value, dtype, input_vector=True)

    def coerce_value(self, value: ValueType, type: DataType) -> tuple[ValueType, DataType]:
        if type.value >= DataType.SHADER or type.value == DataType.GEOMETRY:
            raise TypeError(
                f'Can\'t coerce type {type._name_} to a Geometry Nodes value')
        return value, type

    def resolve_function(self, name: str, args: list[ty_expr], functions: dict[str, list[TyFunction]]) -> tuple[Union[TyFunction, NodeInstance], list[DataType], list[str]]:
        return self._resolve_function(name, args, [geometry_node_aliases, shader_geo_node_aliases], [geometry_nodes, instances, functions])
