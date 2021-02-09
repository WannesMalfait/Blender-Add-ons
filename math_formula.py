

import bpy


bl_info = {
    "name": "Node Math Formula",
    "author": "Wannes Malfait",
    "version": (0, 1, 0),
    "location": "Node Editor Toolbar",
    "description": "Quickly add math nodes by typing in a formula",
    "category": "Node",
    "blender": (2, 91, 0),  # Required so the add-on will actually load
}

# Operations used by the math node
math_operations = [
    # First element contains aliases for the function name
    # Second element is the actual function name
    # Third element is the number of arguments
    (('+', 'add'), 'ADD', 2),
    (('-', 'sub'), 'SUBTRACT', 2),
    (('*', 'mult'), 'MULTIPLY', 2),
    (('/', 'div'), 'DIVIDE', 2),
    (('*+', 'mult_add'), 'MULTIPLY_ADD', 3),
    (('sin', 'sine'), 'SINE', 1),
    (('cos', 'cosine'), 'COSINE', 1),
    (('tan', 'tangent'), 'TANGENT', 1),
    (('asin', 'arcsin', 'arcsine'), 'ARCSINE', 1),
    (('acos', 'arccos', 'arccosine'), 'ARCCOSINE', 1),
    (('atan', 'arctan', 'arctangent'), 'ARCTANGENT', 1),
    (('atan2', 'arctan2'), 'ARCTAN2', 2),
    (('sinh'), 'SINEH', 1),
    (('cosh'), 'COSH', 1),
    (('tanh'), 'TANH', 1),
    (('^', 'pow', 'power'), 'POWER', 2),
    (('log', 'logarithm'), 'LOGARITHM', 2),
    (('sqrt'), 'SQRT', 1),
    (('1/sqrt', 'inv_sqrt'), 'INVERSE_SQRT', 1),
    (('e^x', 'e^', 'exp'), 'EXPONENT', 1),
    (('min', 'minimum'), 'MINIMUM', 2),
    (('max', 'maximum'), 'MAXIMUM', 2),
    (('<', 'less_than'), 'LESS_THAN', 2),
    (('>', 'greater_than'), 'GREATER_THAN', 2),
    (('sgn', 'sign'), 'SIGN', 1),
    (('==', 'compare'), 'COMPARE', 3),
    (('smin', 'smooth_min', 'smooth_minimum'), 'SMOOTH_MIN', 3),
    (('smax', 'smooth_max', 'smooth_maximum'), 'SMOOTH_MAX', 3),
    (('fract'), 'FRACT', 1),
    (('%', 'mod'), 'MODULO', 2),
    (('snap'), 'SNAP', 2),
    (('wrap'), 'WRAP', 3),
    (('pingpong', 'ping_pong'), 'PINGPONG', 2),
    (('abs', 'absolute'), 'ABSOLUTE', 1),
    (('round'), 'ROUND', 1),
    (('floor'), 'FLOOR', 1),
    (('ceil', 'CEIL', 1)),
    (('trunc', 'truncate'), 'TRUNCATE', 1),
    (('rad', 'to_rad', 'to_radians', 'radians'), 'RADIANS', 1),
    (('deg', 'to_deg', 'to_degrees', 'degrees'), 'DEGREES', 1)
]


class MF_Settings(bpy.types.PropertyGroup):
    formula: bpy.props.StringProperty(
        name="Formula",
        description="Formula written in Reverse Polish Notation",
        default="4 5 *",
    )
    temp_attr_name: bpy.props.StringProperty(
        name="Temporary Attribute",
        description="Name of the temporary attribute used to store in between results",
        default="mf_temp",
    )


def mf_check(context):
    space = context.space_data
    return space.type == 'NODE_EDITOR' and space.node_tree is not None and space.tree_type == 'GeometryNodeTree'


class MFBase:
    @classmethod
    def poll(cls, context):
        return mf_check(context)


def get_args(cls, stack, num_args, func_name):
    args = []
    for _ in range(num_args):
        if stack == []:
            cls.report(
                {'WARNING'}, f"Invalid number of arguments for {func_name.lower()}. Expected {num_args} arguments, got args: {args}.")
            args.append("no_arg")
        else:
            args.append(stack.pop())
    args.reverse()
    return args


def is_float(str):
    try:
        float(str)
        return 1
    except:
        return 0


def add_math_node(tree, nodes, args, func_name):
    node = tree.nodes.new(type="GeometryNodeAttributeMath")
    # First node
    if nodes == []:
        node.location = (0, 0)
    else:
        prev_node = nodes[len(nodes)-1]
        node.location = (prev_node.location.x +
                         prev_node.width + 50, prev_node.location.y)
        tree.links.new(prev_node.outputs["Geometry"], node.inputs["Geometry"])
    node.operation = func_name
    l = len(args)
    # 0 -> Attribute 1 -> FLOAT
    arg_types = [is_float(arg) for arg in args]
    # Convert the number strings to floats
    for i in range(l):
        # It's a float
        if arg_types[i] == 1:
            args[i] = float(args[i])
    # Possible types for the socket
    types = ('ATTRIBUTE', 'FLOAT')
    node.input_type_a = types[arg_types[0]]
    if l >= 2:
        node.input_type_b = types[arg_types[1]]
    if l == 3:
        node.input_type_c = types[arg_types[2]]
    for i in range(l):
        # First input is Geometry so we skip it
        # The inputs are in the following order:
        # STRING, FLOAT for each socket
        # So we need to go in pairs of two
        node.inputs[1 + 2*i + arg_types[i]].default_value = args[i]

    nodes.append(node)
    return node


class MF_OT_math_formula_add(bpy.types.Operator, MFBase):
    """Add the nodes for the formula"""
    bl_idname = "node.mf_math_formula_add"
    bl_label = "Add Math Formula"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Safe because of poll function
        tree = context.space_data.node_tree
        props = context.scene.math_formula_add
        # The formula that we parse. Should be in Reverse Polish Notation
        formula = props.formula
        # Used to store temporary results
        temp_attr_name = props.temp_attr_name
        stack = []
        # The nodes that we added
        nodes = []
        for element in formula.split(' '):
            was_func = False
            func_name = None
            args = None
            for operation in math_operations:
                if element in operation[0]:
                    func_name = operation[1]
                    args = get_args(self, stack, operation[2], func_name)
                    was_func = True
                    break
            if was_func:
                node = add_math_node(tree, nodes, args, func_name)
                node.inputs["Result"].default_value = temp_attr_name
                stack.append(temp_attr_name)
            else:
                # It is an argument and not a function
                stack.append(element)
        if nodes != []:
            # Add all nodes in a frame
            frame = tree.nodes.new(type='NodeFrame')
            frame.label = formula
            for node in nodes:
                node.parent = frame
            frame.update()
        # Force an update
        tree.update_tag()
        return {'FINISHED'}


class VF_PT_panel(bpy.types.Panel, MFBase):
    bl_idname = "NODE_PT_mf_math_formula"
    bl_space_type = 'NODE_EDITOR'
    bl_label = "Add Math Formula"
    bl_region_type = "UI"
    bl_category = "Math Formula"

    def draw(self, context):

        # Helper variables
        layout = self.layout
        scene = context.scene
        props = scene.math_formula_add

        col = layout.column(align=True)
        col.prop(props, 'formula')
        col.prop(props, 'temp_attr_name')
        col.separator()
        col.operator(MF_OT_math_formula_add.bl_idname)


classes = (
    MF_Settings,
    MF_OT_math_formula_add,
    VF_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.math_formula_add = bpy.props.PointerProperty(
        type=MF_Settings)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.math_formula_add


if __name__ == "__main__":
    register()
