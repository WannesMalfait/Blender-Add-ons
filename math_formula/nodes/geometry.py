from .base import *

# GEOMETRY


class SeparateGeometry(NodeFunction):
    _name = 'GeometryNodeSeparateGeometry'
    _input_sockets = [
        Socket(0, 'geometry', DataType.GEOMETRY),
        Socket(1, 'selection', DataType.BOOL),
    ]
    _output_sockets = [
        Socket(0, 'selection', DataType.GEOMETRY),
        Socket(1, 'inverted', DataType.GEOMETRY),
    ]
    _props = (
        ('domain', ('POINT', 'EDGE', 'FACE', 'CURVE',),),
    )

    def __init__(self, props) -> None:
        super().__init__(props)
        self.prop_values = [('domain', props[0])]


class SetPosition(NodeFunction):
    _name = 'GeometryNodeSetPosition'
    _input_sockets = [
        Socket(0, 'geometry', DataType.GEOMETRY),
        Socket(1, 'selection', DataType.BOOL),
        Socket(2, 'position', DataType.VEC3),
        Socket(3, 'offset', DataType.VEC3),
    ]
    _output_sockets = [
        Socket(0, 'geometry', DataType.GEOMETRY),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)

# INPUT


class InputNormal(NodeFunction):
    _name = 'GeometryNodeInputNormal'
    _input_sockets = []
    _output_sockets = [
        Socket(0, 'normal', DataType.VEC3),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)


class InputPosition(NodeFunction):
    _name = 'GeometryNodeInputPosition'
    _input_sockets = []
    _output_sockets = [
        Socket(0, 'position', DataType.VEC3),
    ]

# MESH PRIMITIVES


class MeshUVSphere(NodeFunction):
    _name = 'GeometryNodeMeshUVSphere'
    _input_sockets = [
        Socket(0, 'segments', DataType.INT),
        Socket(1, 'rings', DataType.INT),
        Socket(2, 'radius', DataType.FLOAT),
    ]
    _output_sockets = [
        Socket(0, 'mesh', DataType.GEOMETRY),
    ]

    def __init__(self, props) -> None:
        super().__init__(props)


functions = {
    # Geometry
    'separate_geometry': SeparateGeometry,
    'set_position': SetPosition,
    # Input
    'normal': InputNormal,
    'position': InputPosition,
    # Mesh Primitives
    'uv_sphere': MeshUVSphere,
}
