from .base import *


# TEXTURE NODES:

class WhiteNoise(NodeFunction):
    _name = 'ShaderNodeTexWhiteNoise'

    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
        Socket(1, 'w', DataType.FLOAT),
    ]

    _output_sockets = [
        Socket(0, 'value', DataType.FLOAT),
        Socket(1, 'color', DataType.RGBA),
    ]

    _props = (
        ('noise_dimensions', (
            '1D',
            '2D',
            '3D',
            '4D',
        ),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        noise_dimensions = None
        sockets = []
        if len(props) != 1:
            noise_dimensions = '3D'
        else:
            noise_dimensions = props[0]
        if noise_dimensions != '1D':
            sockets = [self._input_sockets[0]]
        if noise_dimensions in ('1D', '4D'):
            sockets.append(self._input_sockets[1])
        self.prop_values = [('noise_dimensions', noise_dimensions)]


# class Voronoi(NodeFunction):
#     _name = 'ShaderNodeTexVoronoi'

#     _input_sockets = [
#         Socket(0, 'vector', DataType.VEC3),
#         Socket(1, 'w', DataType.FLOAT),
#         Socket(2, 'scale', DataType.FLOAT),
#         Socket(3, 'detail', DataType.FLOAT),
#         Socket(4, 'roughness', DataType.FLOAT),
#         Socket(5, 'distortion', DataType.FLOAT),
#     ]

#     _output_sockets = [
#         Socket(0, 'fac', DataType.FLOAT),
#         Socket(1, 'color', DataType.RGBA),
#     ]

#     _props = (
#         ('voronoi_dimensions', (
#             '1D',
#             '2D',
#             '3D',
#             '4D',
#         ),),
#     )

#     def __init__(self, props) -> None:
#         super().__init__(props)
#         voronoi_dimensions = None
#         sockets = []
#         if len(props) != 1:
#             voronoi_dimensions = '3D'
#         else:
#             voronoi_dimensions = props[0]
#         if voronoi_dimensions != '1D':
#             sockets = [self._input_sockets[0]]
#         if voronoi_dimensions in ('1D', '4D'):
#             sockets.append(self._input_sockets[1])
#         sockets += self._input_sockets[2:]
#         self.prop_values = [('voronoi_dimensions', voronoi_dimensions)]


class Noise(NodeFunction):
    _name = 'ShaderNodeTexNoise'

    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
        Socket(1, 'w', DataType.FLOAT),
        Socket(2, 'scale', DataType.FLOAT),
        Socket(3, 'detail', DataType.FLOAT),
        Socket(4, 'roughness', DataType.FLOAT),
        Socket(5, 'distortion', DataType.FLOAT),
    ]

    _output_sockets = [
        Socket(0, 'fac', DataType.FLOAT),
        Socket(1, 'color', DataType.RGBA),
    ]

    _props = (
        ('noise_dimensions', (
            '1D',
            '2D',
            '3D',
            '4D',
        ),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        noise_dimensions = None
        sockets = []
        if len(props) != 1:
            noise_dimensions = '3D'
        else:
            noise_dimensions = props[0]
        if noise_dimensions != '1D':
            sockets = [self._input_sockets[0]]
        if noise_dimensions in ('1D', '4D'):
            sockets.append(self._input_sockets[1])
        sockets += self._input_sockets[2:]
        self.prop_values = [('noise_dimensions', noise_dimensions)]


class Wave(NodeFunction):
    _name = 'ShaderNodeTexWave'

    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
        Socket(1, 'scale', DataType.FLOAT),
        Socket(2, 'distortion', DataType.FLOAT),
        Socket(3, 'detail', DataType.FLOAT),
        Socket(4, 'detail_scale', DataType.FLOAT),
        Socket(5, 'detail_roughness', DataType.FLOAT),
        Socket(6, 'phase_offset', DataType.FLOAT),
    ]

    _output_sockets = [
        Socket(0, 'color', DataType.RGBA),
        Socket(1, 'fac', DataType.FLOAT),
    ]

    _props = (
        ('wave_type', (
            'RINGS',
            'BANDS',
        ),),
        ('bands_direction', (
            'X',
            'Y',
            'Z',
            'DIAGONAL',
        ),),
        ('wave_profile', (
            'SIN',
            'SAW',
            'TRI',
        ),),

    )

    def __init__(self, props) -> None:
        super().__init__(props)
        wave_type = 'RINGS'
        bands_direction = 'X'
        wave_profile = 'SIN'
        if len(props) >= 1:
            wave_type = props[0]
        if len(props) >= 2:
            bands_direction = props[1]
        if len(props) >= 3:
            wave_profile = props[2]
        self.prop_values = [('wave_type', wave_type),
                            ('bands_direction', bands_direction),
                            ('wave_profile', wave_profile), ]


class Gradient(NodeFunction):
    _name = 'ShaderNodeTexGradient'

    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
    ]

    _output_sockets = [
        Socket(0, 'color', DataType.RGBA),
        Socket(1, 'fac', DataType.FLOAT),
    ]

    _props = (
        ('gradient_type', (
            'LINEAR',
            'QUADRATIC',
            'EASING',
            'DIAGONAL',
            'SPHERICAL',
            'QUADRATIC_SPHERE',
            'RADIAL',
        ),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        gradient_type = None
        if len(props) != 1:
            gradient_type = 'LINEAR'
        else:
            gradient_type = props[0]
        self.prop_values = [('gradient_type', gradient_type)]


class Checker(NodeFunction):
    _name = 'ShaderNodeTexChecker'

    _input_sockets = [
        Socket(0, 'vector', DataType.VEC3),
        Socket(1, 'color1', DataType.RGBA),
        Socket(2, 'color2', DataType.RGBA),
        Socket(3, 'scale', DataType.FLOAT),
    ]

    _output_sockets = [
        Socket(0, 'color', DataType.RGBA),
        Socket(1, 'fac', DataType.FLOAT),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)


class Clamp(NodeFunction):
    _name = 'ShaderNodeClamp'
    _input_sockets = [
        Socket(0, 'value', DataType.FLOAT),
        Socket(1, 'min', DataType.FLOAT),
        Socket(2, 'max', DataType.FLOAT),
    ]
    _output_sockets = [
        Socket(0, 'result', DataType.FLOAT)
    ]

    _props = (
        ('clamp_type', (
            'MINMAX',
            'RANGE',
        ),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        clamp_type = None
        if len(props) != 1:
            clamp_type = 'MINMAX'
        else:
            clamp_type = props[0]
        self.prop_values = [('clamp_type', clamp_type)]


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
    # 'voronoi': Voronoi,
    'wave': Wave,
    'white_noise': WhiteNoise,
    'noise': Noise,
    'checker': Checker,
    'gradient': Gradient,
    'clamp': Clamp,
    'combine_xyz': CombineXYZ,
    'separate_xyz': SeparateXYZ,
    'map_range': MapRange,
    'math': Math,
    'vector_math': VectorMath,
}
