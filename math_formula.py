import bpy
import blf
import bgl
import rna_keymap_ui


bl_info = {
    "name": "Node Math Formula",
    "author": "Wannes Malfait",
    "version": (0, 4, 0),
    "location": "Node Editor Toolbar",
    "description": "Quickly add math nodes by typing in a formula",
    "category": "Node",
    "blender": (2, 93, 0),  # Required so the add-on will actually load
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
    (('sinh',), 'SINH', 1),
    (('cosh',), 'COSH', 1),
    (('tanh',), 'TANH', 1),
    (('^', 'pow', 'power'), 'POWER', 2),
    (('log', 'logarithm'), 'LOGARITHM', 2),
    (('sqrt',), 'SQRT', 1),
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
    (('fract',), 'FRACT', 1),
    (('%', 'mod'), 'MODULO', 2),
    (('snap',), 'SNAP', 2),
    (('wrap',), 'WRAP', 3),
    (('pingpong', 'ping_pong'), 'PINGPONG', 2),
    (('abs', 'absolute'), 'ABSOLUTE', 1),
    (('round',), 'ROUND', 1),
    (('floor',), 'FLOOR', 1),
    (('ceil',), 'CEIL', 1),
    (('trunc', 'truncate'), 'TRUNCATE', 1),
    (('rad', 'to_rad', 'to_radians', 'radians'), 'RADIANS', 1),
    (('deg', 'to_deg', 'to_degrees', 'degrees'), 'DEGREES', 1)
]

vector_math_operations = [
    (('v+', 'vadd'), 'ADD', 2),
    (('v-', 'vsub'), 'SUBTRACT', 2),
    (('v*', 'vmult'), 'MULTIPLY', 2),
    (('v/', 'vdiv'), 'DIVIDE', 2),
    (('vcross', 'cross', 'cross_product'), 'CROSS_PRODUCT', 2),
    (('vproject', 'project'), 'PROJECT', 2),
    (('vreflect', 'reflect'), 'REFLECT', 2),
    (('vsnap',), 'SNAP', 2),
    (('v%', 'mod'), 'MODULO', 2),
    (('vmin', 'vminimum'), 'MINIMUM', 2),
    (('vmax', 'vmaximum'), 'MAXIMUM', 2),
    (('vdot', 'dot', 'dot_product'), 'DOT_PRODUCT', 2),
    (('vdist', 'dist', 'distance'), 'DISTANCE', 2),
    (('vlength', 'length',), 'LENGTH', 1),
    # Don't use 'scale' because it's a common attribute name
    (('vscale',), 'SCALE', 2),
    (('vnormalize', 'normalize',), 'NORMALIZE', 1),
    (('vfloor',), 'FLOOR', 1),
    (('vceil',), 'CEIL', 1),
    (('vfract',), 'FRACT', 1),
    (('vabs', 'vabsolute'), 'ABSOLUTE', 1),
    (('vsin', 'vsine'), 'SINE', 1),
    (('vcos', 'vcosine'), 'COSINE', 1),
    (('vtan', 'vtangent'), 'TANGENT', 1),
    (('vwrap',), 'WRAP', 3),
]

builtin_attributes = [
    'position',
    'scale',
    'rotation',
    'radius',
    'id',
]


class MFMathFormula(bpy.types.AddonPreferences):
    bl_idname = __name__

    font_size: bpy.props.IntProperty(
        name="Font Size",
        description="Font size used for displaying text",
        default=15,
        min=8,
        soft_max=20,
    )
    node_distance: bpy.props.IntProperty(
        name="Distance between nodes",
        description="The distance placed between the nodes from the formula",
        default=30,
        min=0,
    )
    builtin_attr_color: bpy.props.FloatVectorProperty(
        name="Built-in Attribute Color",
        default=(0.5, 0.3, 0.05),
        subtype='COLOR',
    )
    math_func_color: bpy.props.FloatVectorProperty(
        name="Math Function Color",
        default=(0.0, 0.8, 0.1),
        subtype='COLOR',
    )
    vector_math_func_color: bpy.props.FloatVectorProperty(
        name="Vector Math Function Color",
        default=(0.142, 0.408, 0.8),
        subtype='COLOR',
    )
    grouping_color: bpy.props.FloatVectorProperty(
        name="Grouping Indicator Color",
        default=(0.3, 0.1, 0.8),
        subtype='COLOR',
    )
    separate_combine_color: bpy.props.FloatVectorProperty(
        name="Combine Separate XYZ Color",
        default=(0.76, 0.195, 0.071),
        subtype='COLOR',
    )
    float_color: bpy.props.FloatVectorProperty(
        name="Number Color",
        default=(0.7, 0.515, 0.462),
        subtype='COLOR',
    )
    default_color: bpy.props.FloatVectorProperty(
        name="Default Color",
        default=(1.0, 1.0, 1.0),
        subtype='COLOR',
    )
    result_color: bpy.props.FloatVectorProperty(
        name="Result Color",
        default=(0.103, 0.8, 0.492),
        subtype='COLOR',
    )

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        col.prop(self, 'font_size')
        col.prop(self, 'node_distance')
        box = layout.box()
        box.label(text="Syntax Highlighting")
        box.prop(self, 'builtin_attr_color')
        box.prop(self, 'math_func_color')
        box.prop(self, 'vector_math_func_color')
        box.prop(self, 'grouping_color')
        box.prop(self, 'separate_combine_color')
        box.prop(self, 'float_color')
        box.prop(self, 'default_color')
        box.prop(self, 'result_color')
        col = layout.column()
        col.label(text="Keymaps:")
        kc = bpy.context.window_manager.keyconfigs.addon
        for km, kmi in addon_keymaps:
            km = km.active()
            col.context_pointer_set("keymap", km)
            rna_keymap_ui.draw_kmi([], kc, km, kmi, col, 0)


class MF_Settings(bpy.types.PropertyGroup):
    formula: bpy.props.StringProperty(
        name="Formula",
        description="Formula written in Reverse Polish Notation",
        default="4 5 *",
    )
    temp_attr_name: bpy.props.StringProperty(
        name="Temporary Attribute",
        description="Name of the temporary attribute used to store in between results",
        default="tmp",
    )
    no_arg: bpy.props.StringProperty(
        name="Missing Argument Name",
        default="",
        description="The name of the attribute used to fill in missing arguments."
    )
    add_frame: bpy.props.BoolProperty(
        name="Add Frame",
        description='Put all the nodes in a frame',
        default=False,
    )


def mf_check(context) -> bool:
    space = context.space_data
    return space.type == 'NODE_EDITOR' and space.node_tree is not None and space.tree_type == 'GeometryNodeTree'


class MFBase:
    @classmethod
    def poll(cls, context):
        return mf_check(context)


class Token():
    """
    Class used to store the possible types of tokens that
    arise from parsing the formula
    """

    def __init__(self, token_type, value, color, print_value=None, data=None) -> None:
        self.type = token_type
        self.data = data
        self.value = value
        self.print_value = print_value
        self.color = color

    def highlight(self, font_id) -> None:
        """
        Set the color for text drawing to the color associated with the token
        """
        blf.color(font_id, self.color[0], self.color[1], self.color[2], 1.0)

    def value_for_print(self) -> str:
        """
        Get the value where the token was constructed from
        """
        return self.print_value if self.print_value is not None else self.value

    def __str__(self) -> str:
        return f"{{Token {self.value_for_print()}: type: {self.type}, value: {self.value}, data: {self.data} }}"


class MFParser:
    """
    Class to parse a formula into a list of tokens.
    """

    def __init__(self) -> None:
        self.tokens = []
        self.reset()

    def reset(self):
        self.current_text = ""
        self.prev_char = " "
        self.making_vector = False
        self.is_res = False

    @staticmethod
    def is_float(input) -> bool:
        """
        Check if the argument can be converted to a `float`.
        """
        try:
            float(input)
            return True
        except:
            return False

    def is_res_check(self, token, prefs):
        if self.is_res and token.type != 'excess':
            if token.type in ('vector', 'combine_xyz'):
                for i in range(3):
                    token.value[i] = str(token.value[i])
            else:
                token.value = token.value_for_print()
            token.type = 'result'
            token.color = prefs.result_color
            self.is_res = False

    def add_token_check(self, prefs) -> None:
        """
        After finishing a token check if it is empty. If not add
        it to the tokens.
        """
        if self.current_text != "" or self.prev_char in "})]":
            # Separate the entries of the vector with a space
            if self.making_vector:
                self.current_text += ' '
            else:
                token = self.get_token(self.current_text)
                self.is_res_check(token, prefs)
                self.tokens.append(token)
                token = Token('excess', ' ', prefs.default_color)
                self.tokens.append(token)
                self.current_text = ""

    def parse(self, string: str, cursor=None) -> list:
        """
        Parse the input string and return a list of tokens
        """
        prefs = bpy.context.preferences.addons[__name__].preferences
        added_cursor = False
        for index, char in enumerate(string):
            if cursor is not None and index == cursor:
                token = Token('cursor', '_', prefs.grouping_color,
                              data=len(self.current_text))
                self.tokens.append(token)
                added_cursor = True
            if char == ' ':
                self.add_token_check(prefs)
            elif char in '[]{}':
                self.add_token_check(prefs)
                token = Token('excess', char, prefs.grouping_color)
                self.tokens.append(token)
            elif char == '(':
                self.add_token_check(prefs)
                # Add it so the user can see what they typed
                token = Token('excess', '(', prefs.default_color)
                self.tokens.append(token)
                self.making_vector = True
            elif char == ')':
                # Add it so the user can see what they typed
                token = self.get_token(self.current_text)
                self.is_res_check(token, prefs)
                self.tokens.append(token)
                token = Token('excess', ')', prefs.default_color)
                self.tokens.append(token)
                self.current_text = ""
                self.making_vector = False
            else:
                self.current_text += char
            self.prev_char = char
        if self.current_text != "":
            token = self.get_token(self.current_text)
            self.is_res_check(token, prefs)
            self.tokens.append(token)
        if cursor is not None and not added_cursor:
            token = Token('cursor', '_', prefs.grouping_color,
                          data=0)
            self.tokens.append(token)
        self.reset()
        return self.tokens

    def get_token(self, string: str) -> Token:
        """
        Create a token from the string and return it
        """
        prefs = bpy.context.preferences.addons[__name__].preferences
        if self.making_vector:
            components = string.split(' ')
            # In case of double spaces
            components = [c for i, c in enumerate(
                components) if c != '' and i < 3]
            l = len(components)
            # Fill in possibly missing components
            components = [components[i] if i < l else 0 for i in range(3)]
            all_floats = True
            for i in range(l):
                if self.is_float(components[i]):
                    components[i] = float(components[i])
                else:
                    all_floats = False
            if all_floats:
                return Token('vector', components,
                             prefs.float_color, print_value=string)
            else:
                return Token('combine_xyz', components, prefs.separate_combine_color, print_value=string)
        elif string == "":
            return Token('excess', '', prefs.default_color)
        elif string == '->':
            self.is_res = True
            return Token('excess', '->', prefs.grouping_color)
        elif (ind := string.find('.')) != -1:
            name = string[:ind]
            end = string[ind:]
            components = ('x' in end, 'y' in end, 'z' in end)
            return Token('separate_xyz', name, prefs.separate_combine_color, print_value=string, data=components)
        else:
            for operation in math_operations:
                if string in operation[0]:
                    return Token(
                        'math_func', operation[1], prefs.math_func_color, print_value=string, data=operation[2])
            for operation in vector_math_operations:
                if string in operation[0]:
                    return Token(
                        'vector_math_func', operation[1], prefs.vector_math_func_color, print_value=string, data=operation[2])
            if self.is_float(string):
                return Token('float', float(string),
                             prefs.float_color, print_value=string)
            elif string in builtin_attributes:
                return Token('default', string, prefs.builtin_attr_color)
            return Token('default', string, prefs.default_color)

    def update_last_token(self, string):
        """
        TODO: Implement this function

        Try updating the last token with the new token from the string
        """
        pass


class MF_OT_math_formula_add(bpy.types.Operator, MFBase):
    """Add the nodes for the formula"""
    bl_idname = "node.mf_math_formula_add"
    bl_label = "Add Math Formula"
    bl_options = {'REGISTER', 'UNDO'}

    use_mouse_location: bpy.props.BoolProperty(
        default=False,
    )

    @staticmethod
    def store_mouse_cursor(context, event):
        space = context.space_data
        tree = space.edit_tree

        # convert mouse position to the View2D for later node placement
        if context.region.type == 'WINDOW':
            # convert mouse position to the View2D for later node placement
            space.cursor_location_from_region(
                event.mouse_region_x, event.mouse_region_y)
        else:
            space.cursor_location = tree.view_center

    def get_args(self, stack, num_args, func_name):
        args = []
        for _ in range(num_args):
            if stack == []:
                self.report(
                    {'WARNING'}, f"Invalid number of arguments for {func_name.lower()}. Expected {num_args} arguments, got args: {args}.")
                args.append(self.no_arg)
            else:
                arg = stack.pop()
                if type(arg) == str and arg.startswith(self.temp_attr_name):
                    self.number_of_temp_attributes = max(
                        0, self.number_of_temp_attributes-1)
                args.append(arg)
        args.reverse()
        return args

    def place_node(self, context, node, nodes):
        prefs = bpy.context.preferences.addons[__name__].preferences
        space = context.space_data
        tree = space.edit_tree
        # First node
        if nodes == []:
            node.location = space.cursor_location if self.use_mouse_location else (
                0, 0)
        else:
            prev_node = nodes[-1]
            node.location = (prev_node.location.x +
                             prev_node.width + prefs.node_distance, prev_node.location.y)
            tree.links.new(
                prev_node.outputs["Geometry"], node.inputs["Geometry"])

    def add_combine_xyz_node(self, context, nodes, vec):
        tree = context.space_data.edit_tree
        node = tree.nodes.new(type="GeometryNodeAttributeCombineXYZ")
        self.place_node(context, node, nodes)
        types = ['FLOAT' if type(
            comp) == float else 'ATTRIBUTE' for comp in vec]
        node.input_type_x = types[0]
        node.input_type_y = types[1]
        node.input_type_z = types[2]
        for i in range(3):
            offset = 0 if types[i] == 'ATTRIBUTE' else 1
            node.inputs[1 + 2*i + offset].default_value = vec[i]
        nodes.append(node)
        return node

    def add_separate_xyz_node(self, context, nodes, name, components):
        tree = context.space_data.edit_tree
        node = tree.nodes.new(type="GeometryNodeAttributeSeparateXYZ")
        self.place_node(context, node, nodes)
        node.input_type = 'ATTRIBUTE'
        node.inputs[1].default_value = name
        vec = ('x', 'y', 'z')
        for i in range(3):
            if components[i]:
                node.inputs[3 + i].default_value = vec[i]
        nodes.append(node)
        return node

    def add_math_node(self, context, nodes, args, func_name):
        tree = context.space_data.edit_tree
        node = tree.nodes.new(type="GeometryNodeAttributeMath")
        self.place_node(context, node, nodes)
        node.operation = func_name
        l = len(args)
        # False -> ATTRIBUTE, True -> FLOAT
        arg_types = ['FLOAT' if type(
            arg) == float else 'ATTRIBUTE' for arg in args]
        # Convert the wrong vectors to strings
        for i in range(l):
            if type(args[i]) == list:
                # It's a vec3
                args[i] = str(args[i])
        # Possible types for the socket
        node.input_type_a = arg_types[0]
        if l >= 2:
            node.input_type_b = arg_types[1]
        if l == 3:
            node.input_type_c = arg_types[2]
        for i in range(l):
            # First input is Geometry so we skip it
            # The inputs are in the following order:
            # Geometry
            # A Attribute
            # A Float
            # B Attribute
            # B Float
            # C Attribute
            # C Float
            # Result Attribute
            # So we need to go in pairs of two
            offset = 0 if arg_types[i] == 'ATTRIBUTE' else 1
            node.inputs[1 + 2*i + offset].default_value = args[i]

        nodes.append(node)
        return node

    def add_vector_math_node(self, context, nodes, args, func_name):
        tree = context.space_data.edit_tree
        node = tree.nodes.new(type="GeometryNodeAttributeVectorMath")
        self.place_node(context, node, nodes)
        node.operation = func_name
        # Socket ordering:
        # Geometry
        # A Attribute
        # A Vector
        # B Attribute
        # B Vector
        # B Float (Only used if func_name is 'SCALE')
        # C Attribute
        # C Vector
        # Result Attribute

        # Index of the first socket for 'A','B', and 'C'
        first_socket = (1, 3, 6)

        l = len(args)

        # If it's a float we convert it to a vec3
        for i in range(l):
            if type(args[i]) == float and not(func_name == 'SCALE' and i == 1):
                args[i] = [args[i] for _ in range(3)]
        arg_types = ['VECTOR' if type(arg) == list else 'FLOAT'if type(
            arg) == float else'ATTRIBUTE' for arg in args]
        node.input_type_a = arg_types[0]
        if l >= 2:
            node.input_type_b = arg_types[1]
        if l == 3:
            node.input_type_c = arg_types[2]

        for i in range(l):
            offset = 0 if arg_types[i] == 'ATTRIBUTE' else 1 if arg_types[i] == 'VECTOR' else 2
            node.inputs[first_socket[i] + offset].default_value = args[i]
        nodes.append(node)
        return node

    def execute(self, context):
        space = context.space_data
        # Safe because of poll function
        tree = space.edit_tree
        props = context.scene.math_formula_add
        # The formula that we parse. Should be in Reverse Polish Notation
        formula = props.formula
        # Used to store temporary results
        self.temp_attr_name = props.temp_attr_name
        # Used to fill in missing arguments
        self.no_arg = props.no_arg
        # If two results are stored as temp attributes we need separate names
        self.number_of_temp_attributes = 0
        stack = []
        # The nodes that we added
        nodes = []
        # Parse the input string into a sequence of tokens
        parser = MFParser()
        tokens = parser.parse(formula)
        for token in tokens:
            if token.type == 'excess':
                continue
            elif token.type == 'math_func' or token.type == 'vector_math_func':
                num_args = token.data
                func_name = token.value
                args = self.get_args(stack, num_args, func_name)
                node = None
                if token.type == 'math_func':
                    node = self.add_math_node(context, nodes, args, func_name)
                else:
                    node = self.add_vector_math_node(
                        context, nodes, args, func_name)
                res_string = self.temp_attr_name + \
                    (str(self.number_of_temp_attributes)
                     if self.number_of_temp_attributes else "")
                node.inputs["Result"].default_value = res_string
                stack.append(res_string)
                self.number_of_temp_attributes += 1
            elif token.type == 'combine_xyz':
                vec = token.value
                node = self.add_combine_xyz_node(context, nodes, vec)
                res_string = self.temp_attr_name + \
                    (str(self.number_of_temp_attributes)
                     if self.number_of_temp_attributes else "")
                node.inputs["Result"].default_value = res_string
                stack.append(res_string)
                self.number_of_temp_attributes += 1
            elif token.type == 'separate_xyz':
                name = token.value
                components = token.data
                node = self.add_separate_xyz_node(
                    context, nodes, name, components)
            elif token.type == 'result':
                if nodes == []:
                    continue
                last_node = nodes[-1]
                if last_node.bl_idname == "GeometryNodeAttributeSeparateXYZ":
                    if type(token.value) != list:
                        token.value = [token.value for _ in range(3)]
                    ind = 0
                    for input_name in ("Result X", "Result Y", "Result Z"):
                        if last_node.inputs[input_name].default_value:
                            last_node.inputs[input_name].default_value = token.value[ind]
                            ind += 1
                else:
                    if type(token.value) == list:
                        token.value = token.value[0]
                    last_node.inputs["Result"].default_value = token.value
            else:
                stack.append(token.value)
        if nodes == [] and stack != []:
            offset = 0
            prefs = bpy.context.preferences.addons[__name__].preferences
            loc = space.cursor_location if self.use_mouse_location else (
                0, 0)
            # Add the given attributes as attribute fill nodes
            for name in stack:
                node = tree.nodes.new('GeometryNodeAttributeFill')
                node.location = (
                    loc[0] + offset, loc[1])
                node.inputs["Attribute"].default_value = str(name)
                offset += node.width + prefs.node_distance
        elif props.add_frame:
            # Add all nodes in a frame
            frame = tree.nodes.new(type='NodeFrame')
            frame.label = formula
            for node in nodes:
                node.parent = frame
            frame.update()
        # Force an update
        tree.nodes.update()
        tree.update_tag()
        return {'FINISHED'}

    def invoke(self, context, event):
        self.store_mouse_cursor(context, event)
        return self.execute(context)


class MF_PT_panel(bpy.types.Panel, MFBase):
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
        col.label(text="Addon Preferences has more settings")
        col.prop(props, 'formula')
        col.prop(props, 'temp_attr_name')
        col.prop(props, 'no_arg')
        col.prop(props, 'add_frame')
        col.separator()
        col.operator(MF_OT_math_formula_add.bl_idname)


def draw_callback_px(self,):
    font_id = 0
    font_size = self.font_size
    blf.size(font_id, font_size, 72)
    # Set the initial positions of the text
    posx = self.mouse_loc[0]
    posy = self.mouse_loc[1]
    posz = 0

    # Get the dimensions so that we know where to place the next text
    width, height = blf.dimensions(font_id, "Formula: ")
    blf.color(font_id, 0.7, 0.0, 0.0, 1.0)
    blf.position(font_id, posx, posy+height+10, posz)
    blf.draw(font_id, "(Press ENTER to confirm, ESC to cancel)")

    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.position(font_id, posx, posy, posz)
    blf.draw(font_id, "Formula: ")
    parser = MFParser()
    tokens = parser.parse(self.formula, cursor=self.cursor_index)
    self.formula = ''
    cursor = None
    cursor_pos = 0
    for index, token in enumerate(tokens):
        if token.type == 'cursor':
            # Find the correct placement of the cursor
            cursor = token
            ind_offset = cursor.data
            self.cursor_index = len(self.formula) + ind_offset
            cursor_pos = width
            if index + 1 < len(tokens):
                next_text = tokens[index+1].value_for_print()
                if next_text != "":
                    cursor_pos += blf.dimensions(font_id,
                                                 next_text[:ind_offset])[0]
            # Draw location of the cursor
            blf.position(font_id, posx + cursor_pos, posy-font_size/10, posz)
            cursor.highlight(font_id)
            blf.draw(font_id, cursor.value_for_print())
            continue

        blf.position(font_id, posx+width, posy, posz)
        token.highlight(font_id)
        text = token.value_for_print()
        self.formula += text
        blf.draw(font_id, text)
        width += blf.dimensions(font_id, text)[0]


class MF_OT_type_formula_then_add_nodes(bpy.types.Operator, MFBase):
    """Type the formula then add the attribute nodes"""
    bl_idname = "node.mf_type_formula_then_add_nodes"
    bl_label = "Type math formula then add node"
    bl_options = {'REGISTER', 'UNDO'}

    def modal(self, context, event):
        context.area.tag_redraw()
        # Exit when they press enter
        if event.type == 'RET':
            bpy.types.SpaceNodeEditor.draw_handler_remove(
                self._handle, 'WINDOW')
            context.scene.math_formula_add.formula = self.formula
            bpy.ops.node.mf_math_formula_add(use_mouse_location=True)
            return {'FINISHED'}
        # Cancel when they press Esc or Rmb
        elif event.type in ('ESC', 'RIGHTMOUSE'):
            bpy.types.SpaceNodeEditor.draw_handler_remove(
                self._handle, 'WINDOW')
            return {'CANCELLED'}

        # Prevent unwanted repetition
        elif event.value == 'RELEASE':
            # Lock is needed because of oversensitve keys
            self.lock = False

        # NAVIGATION
        elif event.type == 'LEFT_ARROW':
            self.cursor_index = max(0, self.cursor_index - 1)
        elif event.type == 'RIGHT_ARROW':
            self.cursor_index = min(len(self.formula), self.cursor_index + 1)
        elif event.type == 'HOME':
            self.cursor_index = 0
        elif event.type == 'END':
            self.cursor_index = len(self.formula)

        # INSERTION + DELETING
        elif (not self.lock or event.is_repeat) and event.type == 'BACK_SPACE' and self.cursor_index != 0:
            # Remove the char at the index
            self.formula = self.formula[:self.cursor_index -
                                        1] + self.formula[self.cursor_index:]
            self.cursor_index = self.cursor_index - 1
            # Prevent over sensitive keys
            self.lock = True
        elif (not self.lock or event.is_repeat) and event.type == 'DEL' and self.cursor_index != len(self.formula):
            # Remove the char at the index + 1
            self.formula = self.formula[:self.cursor_index] + \
                self.formula[self.cursor_index+1:]
            # Prevent wrapping when cursor is at the front
            self.cursor_index = max(0, self.cursor_index - 1)
            self.lock = True
        elif not self.lock and event.ctrl and event.type == 'V':
            # Paste from clipboard
            clipboard = bpy.context.window_manager.clipboard
            # Insert char at the index
            self.formula = self.formula[:self.cursor_index] + \
                clipboard + self.formula[self.cursor_index:]
            self.cursor_index += len(clipboard)
            self.lock = True
        elif event.unicode != "" and event.unicode.isprintable():
            # Only allow printable characters

            # Insert char at the index
            self.formula = self.formula[:self.cursor_index] + \
                event.unicode + self.formula[self.cursor_index:]
            self.cursor_index += 1

        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        args = (self,)
        self._handle = bpy.types.SpaceNodeEditor.draw_handler_add(
            draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        self.mouse_loc = (event.mouse_region_x, event.mouse_region_y)
        self.cursor_index = 0
        self.lock = False
        self.formula = ""
        self.font_size = context.preferences.addons[__name__].preferences.font_size
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


addon_keymaps = []
kmi_defs = [
    # kmi_defs entry: (identifier, key, action, CTRL, SHIFT, ALT, props)
    # props entry: (property name, property value)
    (MF_OT_type_formula_then_add_nodes.bl_idname,
     'F', 'PRESS', False, True, False, None)
]

classes = (
    MFMathFormula,
    MF_Settings,
    MF_OT_math_formula_add,
    MF_PT_panel,
    MF_OT_type_formula_then_add_nodes,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # keymaps (Code from node wrangler)
    addon_keymaps.clear()
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Node Editor', space_type="NODE_EDITOR")
        for (identifier, key, action, CTRL, SHIFT, ALT, props) in kmi_defs:
            kmi = km.keymap_items.new(
                identifier, key, action, ctrl=CTRL, shift=SHIFT, alt=ALT)
            kmi.active = True
            if props:
                for prop, value in props:
                    setattr(kmi.properties, prop, value)
            addon_keymaps.append((km, kmi))

    bpy.types.Scene.math_formula_add = bpy.props.PointerProperty(
        type=MF_Settings)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    del bpy.types.Scene.math_formula_add


if __name__ == "__main__":
    register()
