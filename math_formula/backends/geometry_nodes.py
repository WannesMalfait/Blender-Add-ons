from . import type_defs as td
from .builtin_nodes import geometry_node_aliases, instances, shader_geo_node_aliases
from .main import BackEnd

geometry_nodes: dict[str, list[str]] = {
    "add": ["integer_math_add"],
    "sub": ["integer_math_subtract"],
    "mul": ["integer_math_multiply", "multiply_matrices"],
    "div": ["integer_math_divide"],
    "mul_add": ["integer_math_multiply_add"],
    "pow": ["integer_math_power"],
    "abs": ["integer_math_absolute"],
    "min": ["integer_math_minimum"],
    "max": ["integer_math_maximum"],
    "sign": ["integer_math_sign"],
    "div_round": ["integer_math_divide_round"],
    "div_floor": ["integer_math_divide_floor"],
    "div_ceil": ["integer_math_divide_ceil"],
    "mod": ["integer_math_modulo"],
    "mod_floored": ["integer_math_floored_modulo"],
    "gcd": ["integer_math_greatest_common_divisor"],
    "lcm": ["integer_math_least_common_multiple"],
    "_and": ["boolean_math_and"],
    "_or": ["boolean_math_or"],
    "_not": ["boolean_math_not"],
    "round": ["float_to_integer_round"],
    "floor": ["float_to_integer_floor"],
    "ceil": ["float_to_integer_ceiling"],
    "trunc": ["float_to_integer_truncate"],
    "less_than": [
        "compare_less_than_float_element_wise",
        "compare_less_than_integer_element_wise",
        "compare_color_darker_element_wise",
        "compare_less_than_vector_element_wise",
    ],
    "less_equal": [
        "compare_less_than_or_equal_float_element_wise",
        "compare_less_than_or_equal_integer_element_wise",
        "compare_less_than_or_equal_vector_element_wise",
    ],
    "greater_than": [
        "compare_greater_than_float_element_wise",
        "compare_greater_than_integer_element_wise",
        "compare_color_brighter_element_wise",
        "compare_greater_than_vector_element_wise",
    ],
    "greater_equal": [
        "compare_greater_than_or_equal_float_element_wise",
        "compare_greater_than_or_equal_integer_element_wise",
        "compare_greater_than_or_equal_vector_element_wise",
    ],
    "equal": [
        "compare_equal_float_element_wise",
        "compare_equal_integer_element_wise",
        "compare_equal_vector_element_wise",
        "compare_equal_color_element_wise",
    ],
    "not_equal": [
        "compare_not_equal_float_element_wise",
        "compare_not_equal_integer_element_wise",
        "compare_not_equal_vector_element_wise",
        "compare_not_equal_color_element_wise",
    ],
    "rand": ["random_value_float", "random_value_vector"],
    "rand_int": ["random_value_integer"],
    "rand_bool": ["random_value_boolean"],
}


class GeometryNodesBackEnd(BackEnd):
    def create_input(
        self,
        operations: list[td.Operation],
        name: str,
        value: td.ValueType,
        dtype: td.DataType,
    ):
        return super().create_input_helper(
            operations, name, value, dtype, input_vector=True
        )

    def coerce_value(
        self, value: td.ValueType, type: td.DataType
    ) -> tuple[td.ValueType, td.DataType]:
        if type.value >= td.DataType.SHADER or type.value == td.DataType.GEOMETRY:
            raise TypeError(
                f"Can't coerce type {type._name_} to a Geometry Nodes value"
            )
        return value, type

    def resolve_function(
        self,
        name: str,
        pos_args: list[td.ty_expr],
        keyword_args: list[tuple[str, td.ty_expr]],
        functions: dict[str, list[td.TyFunction]],
    ) -> tuple[
        td.TyFunction | td.NodeInstance, list[td.DataType], list[str], list[int]
    ]:
        return self._resolve_function(
            name,
            pos_args,
            keyword_args,
            [geometry_node_aliases, shader_geo_node_aliases],
            [geometry_nodes, instances, functions],
        )
