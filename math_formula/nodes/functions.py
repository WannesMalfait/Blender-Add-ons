from .base import *


class CombineXYZ(NodeFunction):
    _name = 'ShaderNodeCombineXYZ'
    _input_sockets = [
        Socket(0, 'x', DataType.FLOAT),
        Socket(1, 'y', DataType.FLOAT),
        Socket(2, 'z', DataType.FLOAT),
    ]
    _output_sockets = [Socket(0, 'vector', DataType.VEC3)]

    def __init__(self, props) -> None:
        super().__init__(props)


class SeparateXYZ(NodeFunction):
    _name = 'ShaderNodeSeparateXYZ'
    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
    ]
    _output_sockets = [
        Socket(0, 'x', DataType.FLOAT),
        Socket(1, 'y', DataType.FLOAT),
        Socket(2, 'z', DataType.FLOAT),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)


class MapRange(NodeFunction):
    _name = 'ShaderNodeMapRange'
    _input_sockets = [
        Socket(0, 'value', DataType.FLOAT),
        Socket(1, 'from_min', DataType.FLOAT),
        Socket(2, 'from_max', DataType.FLOAT),
        Socket(3, 'to_min', DataType.FLOAT),
        Socket(4, 'to_max', DataType.FLOAT),
        Socket(5, 'steps', DataType.FLOAT),
    ]
    _output_sockets = [
        Socket(0, 'result', DataType.FLOAT),
    ]

    _props = (
        ('interpolation_type', (
            'LINEAR',
            'STEPPED',
            'SMOOTHSTEP',
            'SMOOTHERSTEP',
        ),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        interpolation_type = None
        if len(props) != 1:
            interpolation_type = 'LINEAR'
        else:
            interpolation_type = props[0]
        if interpolation_type != 'STEPPED':
            self._input_sockets = self._input_sockets[:-1]
        self.prop_values = [('interpolation_type', interpolation_type)]


class Math(NodeFunction):
    _name = 'ShaderNodeMath'
    _input_sockets = [
        Socket(0, 'value1', DataType.FLOAT),
        Socket(1, 'value2', DataType.FLOAT),
        Socket(2, 'value3', DataType.FLOAT),
    ]
    _output_sockets = [
        Socket(0, 'value', DataType.FLOAT),
    ]
    _props = (
        ('operation', (
            'ADD',
            'SUBTRACT',
            'MULTIPLY',
            'DIVIDE',
            'MULTIPLY_ADD',
            'POWER',
            'LOGARITHM',
            'SQRT',
            'INVERSE_SQRT',
            'ABSOLUTE',
            'EXPONENT',
            'MINIMUM',
            'MAXIMUM',
            'LESS_THAN',
            'GREATER_THAN',
            'SIGN',
            'COMPARE',
            'SMOOTH_MIN',
            'SMOOTH_MAX',
            'ROUND',
            'FLOOR',
            'CEIL',
            'TRUNC',
            'FRACT',
            'MODULO',
            'WRAP',
            'SNAP',
            'PINGPONG',
            'SINE',
            'COSINE',
            'TANGENT',
            'ARCSINE',
            'ARCCOSINE',
            'ARCTANGENT',
            'ARCTAN2',
            'SINH',
            'COSH',
            'TANH',
            'RADIANS',
            'DEGREES',
        ),),
    )

    # The operations which have a vector math equivalent
    # The dictionary gives the operation of the equivalent
    # form
    _overloadable = {
        'ADD': 'ADD',
        'SUBTRACT': 'SUBTRACT',
        'MULTIPLY': 'MULTIPLY',
        'DIVIDE': 'DIVIDE',
        'MULTIPLY_ADD': 'MULTIPLY_ADD',
        'ABSOLUTE': 'ABSOLUTE',
        'MINIMUM': 'MINIMUM',
        'MAXIMUM': 'MAXIMUM',
        'FLOOR': 'FLOOR',
        'CEIL': 'CEIL',
        'FRACT': 'FRACTION',
        'MODULO': 'MODULO',
        'WRAP': 'WRAP',
        'SNAP': 'SNAP',
        'SINE': 'SINE',
        'COSINE': 'COSINE',
        'TANGENT': 'TANGENT',
    }

    def __init__(self, props) -> None:
        sockets = [self._input_sockets[0]]
        operation = None
        if len(props) != 1:
            operation = 'ADD'
        else:
            operation = props[0]
        if not operation in (
            'SQRT',
            'SIGN',
            'CEIL',
            'SINE',
            'ROUND',
            'FLOOR',
            'COSINE',
            'ARCSINE',
            'TANGENT',
            'ABSOLUTE',
            'RADIANS',
            'DEGREES',
            'FRACTION',
            'ARCCOSINE',
            'ARCTANGENT',
            'INV_SQRT',
            'TRUNC',
            'EXPONENT',
            'COSH',
            'SINH',
            'TANH',
        ):
            sockets.append(self._input_sockets[1])
        if operation in ('COMPARE',
                         'MULTIPLY_ADD',
                         'WRAP',
                         'SMOOTH_MIN',
                         'SMOOTH_MAX'):
            sockets.append(self._input_sockets[2])
        self._input_sockets = sockets
        self.prop_values = [('operation', operation)]


class VectorMath(NodeFunction):
    _name = 'ShaderNodeVectorMath'
    _input_sockets = [
        Socket(0, 'vector1', DataType.VEC3),
        Socket(1, 'vector2', DataType.VEC3),
        Socket(2, 'vector3', DataType.VEC3),
        Socket(3, 'scale', DataType.FLOAT),
    ]
    _output_sockets = [
        Socket(0, 'vector', DataType.VEC3),
        Socket(1, 'value', DataType.FLOAT),
    ]
    _props = (
        ('operation', (
            'ADD',
            'SUBTRACT',
            'MULTIPLY',
            'DIVIDE',
            'MULTIPLY_ADD',
            'CROSS_PRODUCT',
            'PROJECT',
            'REFLECT',
            'REFRACT',
            'FACEFORWARD',
            'DOT_PRODUCT',
            'DISTANCE',
            'LENGTH',
            'SCALE',
            'ABSOLUTE',
            'MINIMUM',
            'MAXIMUM',
            'FLOOR',
            'CEIL',
            'FRACTION',
            'MODULO',
            'WRAP',
            'SNAP',
            'SINE',
            'COSINE',
            'TANGENT',
        ),),
    )

    def __init__(self, props) -> None:
        sockets = [self._input_sockets[0]]
        operation = None
        if len(props) != 1:
            operation = 'ADD'
        else:
            operation = props[0]
        if not operation in (
            'SINE',
            'COSINE',
            'TANGENT',
            'CEIL',
            'SCALE',
            'FLOOR',
            'LENGTH',
            'ABSOLUTE',
            'FRACTION',
            'NORMALIZE',
        ):
            sockets.append(self._input_sockets[1])
        if operation in ('FACEFORWARD',
                         'MULTIPLY_ADD',
                         'WRAP',):
            sockets.append(self._input_sockets[2])
        if operation == 'SCALE':
            sockets.append(self._input_sockets[3])
        self._input_sockets = sockets
        float_socket = operation in ('LENGTH', 'DISTANCE', 'DOT_PRODUCT')
        if float_socket:
            self._output_sockets = self._output_sockets[1:]
        else:
            self._output_sockets = self._output_sockets[:1]
        self.prop_values = [('operation', operation)]


functions = {
    'combine_xyz': CombineXYZ,
    'separate_xyz': SeparateXYZ,
    'map_range': MapRange,
    'math': Math,
    'vector_math': VectorMath,
}
