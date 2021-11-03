from .base import *


# Input

class TexCoord(NodeFunction):
    _name = 'ShaderNodeTexCoord'
    _input_sockets = []
    _output_sockets = [
        Socket(0, 'generated', DataType.VEC3),
        Socket(1, 'normal', DataType.VEC3),
        Socket(2, 'uv', DataType.VEC3),
        Socket(3, 'object', DataType.VEC3),
        Socket(4, 'camera', DataType.VEC3),
        Socket(5, 'window', DataType.VEC3),
        Socket(6, 'reflection', DataType.VEC3),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)


functions = {
    'tex_coord': TexCoord,
}
