from .base import *


class CombineXYZ(NodeFunction):
    _name = 'CombineXYZ'
    _input_sockets = [
        Socket(0, 'x', DataType.FLOAT),
        Socket(1, 'y', DataType.FLOAT),
        Socket(2, 'z', DataType.FLOAT),
    ]
    _output_sockets = [Socket(0, 'vector', DataType.VEC3)]

    def __init__(self, props) -> None:
        super().__init__(props)


class Math(NodeFunction):
    _name = 'Math'
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

    def __init__(self, props) -> None:
        sockets = [self._input_sockets[0]]
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
    _name = 'VectorMath'
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
        print(self._output_sockets)
        self.prop_values = [('operation', operation)]


functions = {
    'combine_xyz': CombineXYZ,
    'math': Math,
    'vector_math': VectorMath,
}
