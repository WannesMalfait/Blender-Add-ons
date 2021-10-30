from bpy.types import NodeSocket
from .parser import Compiler, Error, InstructionType, string_to_data_type
from .scanner import TokenType
from .positioning import TreePositioner
from .nodes.base import DataType, NodeFunction, Value
import time
import bpy
import blf
import os


formula_history = []

font_directory = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), 'fonts')
fonts = {
    'bfont': 0,
    'regular': blf.load(os.path.join(font_directory, 'Anonymous_Pro.ttf')),
    'italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_I.ttf')),
    'bold': blf.load(os.path.join(font_directory, 'Anonymous_Pro_0.ttf')),
    'bold_italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_BI.ttf')),
}


def mf_check(context) -> bool:
    space = context.space_data
    possible_trees = ('GeometryNodeTree', 'ShaderNodeTree')
    return space.type == 'NODE_EDITOR' and space.node_tree is not None and \
        space.tree_type in possible_trees


class MFBase:
    @classmethod
    def poll(cls, context):
        return mf_check(context)


class MF_OT_select_from_root(bpy.types.Operator):
    """Select nodes linked to the active node """
    bl_idname = "node.mf_select_from_root"
    bl_label = "Select nodes from active"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(self, context) -> bool:
        return mf_check(context) and context.active_node is not None

    select_parents: bpy.props.BoolProperty(
        name="Select Parents",
        description="Select all the parents of this node recursively",
        default=False,
    )
    select_children: bpy.props.BoolProperty(
        name="Select Children",
        description="Select all the children of this node recursively",
        default=False,
    )

    def select_parents_of(self, node: bpy.types.Node, links: list[bpy.types.NodeLink], visited: list[bpy.types.Node]) -> None:
        # Prevent loops
        if node in visited:
            return
        node.select = True
        visited.append(node)
        if not node.outputs:
            return
        for link in links:
            if link.from_node == node:
                self.select_parents_of(link.to_node, links, visited)

    def select_children_of(self, node: bpy.types.Node, links: list[bpy.types.NodeLink], visited: list[bpy.types.Node]) -> None:
        # Prevent loops
        if node in visited:
            return
        node.select = True
        visited.append(node)
        if not node.inputs:
            return
        for link in links:
            if link.to_node == node:
                self.select_children_of(link.from_node, links, visited)

    def execute(self, context: bpy.types.Context):
        space = context.space_data
        links = space.edit_tree.links
        active_node = context.active_node
        bpy.ops.node.select_all(action='DESELECT')
        active_node.select = True
        if self.select_children:
            self.select_children_of(active_node, links, [])
        if self.select_parents:
            self.select_parents_of(active_node, links, [])
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class MF_OT_arrange_from_root(bpy.types.Operator):
    """Arange the nodes in the tree with the active node as root"""
    bl_idname = "node.mf_arrange_from_root"
    bl_label = "Arrange nodes from root"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(self, context) -> bool:
        return mf_check(context) and context.active_node is not None

    def execute(self, context: bpy.types.Context):
        space = context.space_data
        links = space.edit_tree.links
        active_node = context.active_node
        bpy.ops.node.select_all(action='DESELECT')
        active_node.select = True
        # Figure out the parents, children, and siblings of nodes.
        # Needed for the node positioner
        node_positioner = TreePositioner(context)
        node_positioner.place_nodes(active_node, links)
        return {'FINISHED'}

    def invoke(self, context, event):
        return self.execute(context)


class MF_OT_math_formula_add(bpy.types.Operator, MFBase):
    """Add the math nodes for the formula"""
    bl_idname = "node.mf_math_formula_add"
    bl_label = "Math Formula"
    bl_options = {'REGISTER', 'UNDO'}

    use_mouse_location: bpy.props.BoolProperty(
        default=False,
    )

    @ staticmethod
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

    def get_args(self, stack: list, num_args: int) -> list[Value]:
        args = []
        for _ in range(num_args):
            arg = stack.pop()
            args.append(arg)
        args.reverse()
        return args

    @staticmethod
    def add_func(context: bpy.context, args: list[Value], function: NodeFunction):
        tree: bpy.types.NodeTree = context.space_data.edit_tree
        if tree.type == 'GeometryNodeTree':
            node = tree.nodes.new(type="GeometryNode" + function.name())
        else:
            node = tree.nodes.new(type="ShaderNode" + function.name())
        for name, value in function.props():
            setattr(node, name, value)
        for i, socket in enumerate(function.input_sockets()):
            arg = args[i]
            if isinstance(arg.value, bpy.types.NodeSocket):
                tree.links.new(arg.value, node.inputs[socket.index])
            elif not arg.value is None:
                node.inputs[i].default_value = arg.value
        return node

    @staticmethod
    def get_value_as_socket(value, name: str, tree) -> tuple:
        if isinstance(value, float):
            node = tree.nodes.new('ShaderNodeValue')
            node.label = name
            node.outputs[0].default_value = value
            socket = node.outputs[0]
            return socket
        elif isinstance(value, list):
            node = tree.nodes.new('ShaderNodeCombineXYZ')
            node.label = name
            for i in range(3):
                node.inputs[i].default_value = value[i]
            socket = node.outputs[0]
            return socket
        else:
            return value

    @staticmethod
    def separate_xyz(value, names, stack, variables, tree) -> tuple:
        # position.xyz; theta = atan2(y,x); r = length({x,y,0}); {r,theta,z}
        out_socket = value
        node = tree.nodes.new('ShaderNodeSeparateXYZ')
        tree.links.new(node.inputs[0], out_socket)
        last_set = None
        for i, component in enumerate(names):
            if component == '':
                continue
            else:
                socket = node.outputs[i]
                variables[component] = socket
                last_set = socket
        if last_set is None:
            stack.append(0)
        else:
            stack.append(last_set)

    def execute(self, context: bpy.context):
        space = context.space_data
        # Safe because of poll function
        tree: bpy.types.NodeTree = space.edit_tree
        props = context.scene.math_formula_add
        # The formula that we parse. Should be in Reverse Polish Notation
        formula = props.formula
        stack = []
        # The nodes that we added
        nodes = []
        root_nodes = []
        # Variables in the form of output sockets
        variables = {}
        # Parse the input string into a sequence of tokens
        compiler = Compiler()
        success = compiler.compile(formula)
        if not success:
            return {'CANCELLED'}
        instructions = compiler.instructions
        for instruction in instructions:
            instruction_type = instruction.instruction
            data = instruction.data
            if instruction_type == InstructionType.VALUE:
                stack.append(data)
            elif instruction_type == InstructionType.FUNCTION:
                function: NodeFunction = data
                print(function.input_sockets())
                args = self.get_args(stack, len(function.input_sockets()))
                node = self.add_func(context, args, function)
                for socket in function.output_sockets():
                    stack.append(
                        Value(socket.sock_type, node.outputs[socket.index]))
                nodes.append(node)
            elif instruction_type == InstructionType.END_OF_STATEMENT:
                if stack == []:
                    print('EMPTY STACK!!!!')
                    continue
                element = stack.pop()
                if isinstance(element, Value) and isinstance(element.value, NodeSocket):
                    root_nodes.append(element.value.node)
            else:
                print(f'Need implementation of {instruction}')
                raise NotImplementedError
        if props.add_frame and nodes != []:
            # Add all nodes in a frame
            frame = tree.nodes.new(type='NodeFrame')
            frame.label = formula
            for node in nodes:
                node.parent = frame
            frame.update()
        self.root_nodes = []
        if root_nodes != []:
            for root_node in root_nodes:
                invalid = False
                for socket in root_node.outputs:
                    # It was connected later on, and is not a root node anymore
                    if socket.is_linked:
                        invalid = True
                        break
                if invalid:
                    continue
                else:
                    self.root_nodes.append(root_node)
        return {'FINISHED'}

    def modal(self, context, event):
        if self.root_nodes[0].dimensions.x == 0:
            return {'RUNNING_MODAL'}
        space = context.space_data
        links = space.edit_tree.links
        cursor_loc = space.cursor_location if self.use_mouse_location else (
            0, 0)
        node_positioner = TreePositioner(context)
        cursor_loc = node_positioner.place_nodes(
            self.root_nodes, links, cursor_loc=cursor_loc)

        return {'FINISHED'}

    def invoke(self, context, event):
        self.store_mouse_cursor(context, event)
        self.execute(context)
        if self.root_nodes == []:
            return {'FINISHED'}
        else:
            # Hacky way to force an update such that node dimensions are correct
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}


class MF_OT_attribute_math_formula_add(bpy.types.Operator, MFBase):
    """Add the attribute nodes for the formula"""
    bl_idname = "node.mf_attribute_math_formula_add"
    bl_label = "Attribute Math Formula"
    bl_options = {'REGISTER', 'UNDO'}

    use_mouse_location: bpy.props.BoolProperty(
        default=False,
    )

    @ staticmethod
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

    def get_args(self, stack, num_args):
        args = []
        for _ in range(num_args):
            arg = stack.pop()
            if isinstance(arg, str) and self.temp_attr_name in arg:
                self.number_of_temp_attributes = max(
                    0, self.number_of_temp_attributes-1)
            args.append(arg)
        args.reverse()
        return args

    def place_node(self, context, node, nodes):
        prefs = bpy.context.preferences.addons['math_formula'].preferences
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
            if type(args[i]) == float and not ((func_name == 'SCALE' and i == 1) or (func_name == 'REFRACT' and i == 2)):
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
        compiler = Compiler()
        success = compiler.compile(formula)
        if not success:
            return {'CANCELLED'}
        instructions = compiler.instructions
        for instruction in instructions:
            instruction_type = instruction.instruction
            data = instruction.data
            if instruction_type == InstructionType.NUMBER:
                stack.append(float(data))
            elif instruction_type == InstructionType.MISSING_ARG:
                stack.append(self.no_arg)
            elif instruction_type in (InstructionType.MATH_FUNC, InstructionType.VECTOR_MATH_FUNC):
                func_name, num_args = data
                args = self.get_args(stack, num_args)
                node = None
                if instruction_type == InstructionType.MATH_FUNC:
                    node = self.add_math_node(
                        context, nodes, args, func_name)
                else:
                    node = self.add_vector_math_node(
                        context, nodes, args, func_name)
                res_string = self.temp_attr_name + \
                    (str(self.number_of_temp_attributes)
                     if self.number_of_temp_attributes else "")
                node.inputs["Result"].default_value = res_string
                stack.append(res_string)
                self.number_of_temp_attributes += 1
            elif instruction_type == InstructionType.END_OF_STATEMENT:
                if stack == []:
                    print('EMPTY STACK!!!!')
                    continue
                stack.pop()
            elif instruction_type == InstructionType.MAKE_VECTOR:
                args = self.get_args(stack, 3)
                all_float = True
                for arg in args:
                    if not isinstance(arg, float):
                        all_float = False
                        break
                if all_float:
                    stack.append(args)
                    continue
                # It contains attributes, so we need a Combine XYZ
                node = self.add_combine_xyz_node(context, nodes, args)
                res_string = self.temp_attr_name + \
                    (str(self.number_of_temp_attributes)
                     if self.number_of_temp_attributes else "")
                node.inputs["Result"].default_value = res_string
                stack.append(res_string)
                self.number_of_temp_attributes += 1
            elif instruction_type == InstructionType.ATTRIBUTE:
                stack.append(data)
            elif instruction_type == InstructionType.VAR:
                stack.append(data)
            elif instruction_type == InstructionType.VECTOR_VAR:
                stack.append(data)
            elif instruction_type == InstructionType.DEFINE:
                name, result = self.get_args(stack, 2)
                if isinstance(result, float) or isinstance(result, list):
                    node = tree.nodes.new('GeometryNodeAttributeFill')
                    self.place_node(context, node, nodes)
                    node.inputs['Attribute'].default_value = str(name)
                    stack.append(str(name))
                    if isinstance(result, float):
                        # Float is default data type
                        node.inputs[3].default_value = result
                    else:
                        node.data_type = 'FLOAT_VECTOR'
                        node.inputs[2].default_value = result
                    nodes.append(node)
                elif isinstance(name, tuple):
                    self.add_separate_xyz_node(
                        context, nodes, result, ('x', 'y', 'z'))
                    # Separate XYZ
                    last_node = nodes[-1]
                    if last_node.bl_idname == 'GeometryNodeAttributeSeparateXYZ':
                        for i, res_name in enumerate(('Result X', 'Result Y', 'Result Z')):
                            last_node.inputs[res_name].default_value = name[i]
                    best_name = ''
                    for comp in name:
                        if comp != '':
                            best_name = comp
                    stack.append(best_name)
                elif nodes != []:
                    last_node = nodes[-1]
                    if last_node.bl_idname == 'GeometryNodeAttributeSeparateXYZ':
                        for res_name in ('Result X', 'Result Y', 'Result Z'):
                            last_node.inputs[res_name].default_value = name
                    elif last_node.bl_idname == 'GeometryNodeAttributeFill':
                        last_node.inputs['Attribute'].default_value = name
                    else:
                        last_node.inputs['Result'].default_value = name
                    stack.append(name)
                else:
                    stack.append(name)
            elif instruction_type == InstructionType.SEPARATE:
                name, data = self.get_args(stack, 2)
                components = [
                    a if a in data else None for a in ('x', 'y', 'z')]
                node = self.add_separate_xyz_node(
                    context, nodes, name, components)
                best_name = 'x'
                for comp in components:
                    if comp is not None:
                        best_name = comp
                stack.append(best_name)
        if props.add_frame and nodes != []:
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


def draw_callback_px(self, context: bpy.context):
    prefs = context.preferences.addons['math_formula'].preferences
    font_id = fonts['regular']
    font_size = prefs.font_size
    blf.size(font_id, font_size, 72)
    # show_errors = time.time()-self.last_action > 2
    # show_errors = False
    # Set the initial positions of the text
    posx = self.formula_loc[0]
    posy = self.formula_loc[1]
    posz = 0

    # Get the dimensions so that we know where to place the next text
    width, height = blf.dimensions(font_id, "Formula: ")
    # Color for the non-user text.
    blf.color(font_id, 0.4, 0.5, 0.1, 1.0)
    blf.position(font_id, posx, posy+height+10, posz)
    blf.draw(font_id, "(Press ENTER to confirm, ESC to cancel)")

    blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
    blf.position(font_id, posx, posy, posz)
    blf.draw(font_id, "Formula: ")
    if len(formula_history) >= self.formula_history_loc > 0:
        self.formula = formula_history[-self.formula_history_loc]
        self.cursor_index = len(self.formula)
    formula = self.formula
    tokens = Compiler.get_tokens(formula)
    cursor_index = self.cursor_index
    cursor_pos_set = False
    cursor_pos = width
    prev = 0
    for i, token in enumerate(tokens):
        blf.position(font_id, posx+width, posy, posz)
        text = token.lexeme
        start = token.start
        # Get cursor relative offset
        if not cursor_pos_set and start >= cursor_index:
            cursor_pos = width - blf.dimensions(
                font_id, formula[cursor_index:start])[0]
            cursor_pos_set = True
        # Draw white space
        white_space = formula[prev:start]
        blf.draw(font_id, white_space)
        width += blf.dimensions(font_id, white_space)[0]
        token_font_style = font_id
        prev_token = tokens[i-1] if i > 0 else token
        if token.token_type == TokenType.IDENTIFIER and prev_token.token_type == TokenType.COLON:
            # Check if it's a valid type
            color(token_font_style,
                  prefs.type_color if token.lexeme in string_to_data_type else prefs.default_color)
        elif TokenType.LET.value <= token.token_type.value <= TokenType.FALSE.value:
            color(token_font_style, prefs.keyword_color)
        elif token.token_type in (TokenType.INT, TokenType.FLOAT):
            color(token_font_style, prefs.number_color)
        elif token.token_type == TokenType.PYTHON:
            token_font_style = fonts['bold']
            color(token_font_style, prefs.python_color)
        elif token.token_type == TokenType.ERROR:
            text, error = token.lexeme
            token_font_style = fonts['italic']
            color(token_font_style, prefs.error_color)
        elif token.token_type == TokenType.STRING:
            color(token_font_style, prefs.string_color)
        else:
            color(token_font_style, prefs.default_color)
        blf.size(token_font_style, font_size, 72)

        blf.position(token_font_style, posx+width, posy, posz)
        blf.draw(token_font_style, text)
        width += blf.dimensions(token_font_style, text)[0]
        prev = start + len(text)

    # Remaining white space at the end
    width += blf.dimensions(font_id, formula[prev:])[0]

    # Errors
    color(font_id, prefs.error_color)
    for n, error in enumerate(self.errors):
        blf.position(font_id, posx, posy-10-font_size*(n+1), posz)
        blf.draw(font_id, error.message)
    # Cursor is in the last token
    if not cursor_pos_set and tokens != []:
        cursor_pos = width-blf.dimensions(font_id,
                                          formula[cursor_index:])[0]
    # Draw cursor
    blf.color(font_id, 0.1, 0.4, 0.7, 1.0)
    blf.position(font_id, posx+cursor_pos, posy-font_size/10, posz)
    blf.draw(font_id, '_')


def color(font_id, color):
    blf.color(font_id, color[0], color[1], color[2], 1.0)


class MF_OT_type_formula_then_add_nodes(bpy.types.Operator, MFBase):
    """Type the formula then add the nodes"""
    bl_idname = "node.mf_type_formula_then_add_nodes"
    bl_label = "Type math formula then add nodes"
    bl_options = {'REGISTER'}

    def modal(self, context: bpy.context, event: bpy.types.Event):
        context.area.tag_redraw()
        action = False
        # Exit when they press enter
        if event.type == 'RET':
            compiler = Compiler()
            res = compiler.compile(self.formula)
            if not res:
                self.errors = compiler.errors
                self.report(
                    {'WARNING'}, 'Compile errors, could not create node tree')
                return {'RUNNING_MODAL'}
            # bpy.types.SpaceNodeEditor.draw_handler_remove(
            #     self._handle, 'WINDOW')
            # print(compiler.instructions)
            # return {'FINISHED'}
            bpy.types.SpaceNodeEditor.draw_handler_remove(
                self._handle, 'WINDOW')
            context.scene.math_formula_add.formula = self.formula
            formula_history.append(self.formula)
            # Deselect all the nodes before adding new ones
            bpy.ops.node.select_all(action='DESELECT')
            bpy.ops.node.mf_math_formula_add(
                'INVOKE_DEFAULT', use_mouse_location=True)
            return {'FINISHED'}
        # Cancel when they press Esc or Rmb
        elif event.type in ('ESC', 'RIGHTMOUSE'):
            bpy.types.SpaceNodeEditor.draw_handler_remove(
                self._handle, 'WINDOW')
            return {'CANCELLED'}

        # Prevent unwanted repetition
        elif event.value == 'RELEASE':
            # Lock is needed because of oversensitive keys
            self.lock = False
            self.middle_mouse = False

        # Compile and check for errors
        elif not self.lock and event.alt and event.type == 'C':
            compiler = Compiler()
            res = compiler.compile(self.formula)
            self.errors = compiler.errors
            if res:
                self.report({'INFO'}, 'No errors detected')
            else:
                self.report({'WARNING'}, 'Compilation failed')
                print('Instructions: ', compiler.instructions)

        # NAVIGATION
        elif event.type == 'MIDDLEMOUSE':
            self.old_mouse_loc = (event.mouse_region_x, event.mouse_region_y)
            self.old_formula_loc = self.formula_loc
            self.middle_mouse = True
        elif event.type == 'MOUSEMOVE' and self.middle_mouse:
            self.formula_loc = (
                self.old_formula_loc[0] +
                event.mouse_region_x - self.old_mouse_loc[0],
                self.old_formula_loc[1] +
                event.mouse_region_y - self.old_mouse_loc[1])
        elif event.type == 'WHEELUPMOUSE':
            prefs = context.preferences.addons['math_formula'].preferences
            prefs.font_size += 1
        elif event.type == 'WHEELDOWNMOUSE':
            prefs = context.preferences.addons['math_formula'].preferences
            prefs.font_size = max(8, prefs.font_size-1)

        # CURSOR NAVIGATION
        elif event.type == 'LEFT_ARROW':
            self.cursor_index = max(0, self.cursor_index - 1)
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif event.type == 'RIGHT_ARROW':
            self.cursor_index = min(len(self.formula), self.cursor_index + 1)
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif event.type == 'HOME':
            self.cursor_index = 0
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif event.type == 'END':
            self.cursor_index = len(self.formula)
            # We are now editing this one
            self.formula_history_loc = 0
            action = True

        # FORMULA HISTORY
        elif event.type == 'UP_ARROW':
            self.formula_history_loc = min(
                len(formula_history), self.formula_history_loc + 1)
            action = True
        elif event.type == 'DOWN_ARROW':
            self.formula_history_loc = max(0, self.formula_history_loc - 1)
            action = True

        # INSERTION + DELETING
        elif (not self.lock or event.is_repeat) and event.type == 'BACK_SPACE' and self.cursor_index != 0:
            # Remove the char at the index
            self.formula = self.formula[:self.cursor_index -
                                        1] + self.formula[self.cursor_index:]
            self.cursor_index = self.cursor_index - 1
            # Prevent over sensitive keys
            self.lock = True
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif (not self.lock or event.is_repeat) and event.type == 'DEL' and self.cursor_index != len(self.formula):
            # Remove the char at the index + 1
            self.formula = self.formula[:self.cursor_index] + \
                self.formula[self.cursor_index+1:]
            # Prevent wrapping when cursor is at the front
            self.cursor_index = max(0, self.cursor_index - 1)
            self.lock = True
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif not self.lock and event.ctrl and event.type == 'V':
            # Paste from clipboard
            clipboard = context.window_manager.clipboard
            # Insert char at the index
            self.formula = self.formula[:self.cursor_index] + \
                clipboard + self.formula[self.cursor_index:]
            self.cursor_index += len(clipboard)
            self.lock = True
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        elif event.unicode != "" and event.unicode.isprintable():
            # Only allow printable characters

            # Insert char at the index
            self.formula = self.formula[:self.cursor_index] + \
                event.unicode + self.formula[self.cursor_index:]
            self.cursor_index += 1
            # We are now editing this one
            self.formula_history_loc = 0
            action = True
        if action:
            self.last_action = time.time()
        return {'RUNNING_MODAL'}

    def invoke(self, context: bpy.context, event: bpy.types.Event):
        args = (self, context)
        self._handle = bpy.types.SpaceNodeEditor.draw_handler_add(
            draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        self.formula_loc = (event.mouse_region_x, event.mouse_region_y)
        # Stores the location of the formula before dragging MMB
        self.old_formula_loc = self.formula_loc
        self.old_mouse_loc = (0, 0)
        self.cursor_index = 0
        self.lock = False
        self.middle_mouse = False
        self.formula = ""
        self.formula_history_loc = 0
        self.errors: list[Error] = []
        self.last_action = time.time()
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


classes = (
    MF_OT_arrange_from_root,
    MF_OT_select_from_root,
    MF_OT_math_formula_add,
    MF_OT_type_formula_then_add_nodes,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
