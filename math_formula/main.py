from .nodes.base import DataType, NodeFunction, Value, ValueType
import time
import bpy
import blf
from . import file_loading
from .file_loading import fonts
from .positioning import TreePositioner
from .scanner import TokenType
from .parser import Compiler, Error, InstructionType, OpType, string_to_data_type
from bpy.types import NodeSocket

formula_history = []


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
        args = stack[-num_args:]
        stack[:] = stack[:-num_args]
        return args

    @staticmethod
    def add_func(context: bpy.context, args: list[ValueType], function: NodeFunction):
        tree: bpy.types.NodeTree = context.space_data.edit_tree
        if tree.type == 'GeometryNodeTree':
            node = tree.nodes.new(type="GeometryNode" + function.name())
        else:
            node = tree.nodes.new(type="ShaderNode" + function.name())
        for name, value in function.props():
            setattr(node, name, value)
        for i, socket in enumerate(function.input_sockets()):
            arg = args[i]
            if isinstance(arg, bpy.types.NodeSocket):
                tree.links.new(arg, node.inputs[socket.index])
            elif not arg is None:
                node.inputs[socket.index].default_value = arg
        return node

    @staticmethod
    def get_value_as_socket(value: ValueType, type: DataType, tree: bpy.types.NodeTree) -> tuple[bpy.types.Node, NodeSocket]:
        node = None
        node_prefix = 'ShaderNode'
        if type == DataType.UNKNOWN or type == DataType.DEFAULT or type == DataType.FLOAT:
            node = tree.nodes.new(node_prefix + 'Value')
            if value is not None:
                node.outputs[0].default_value = value
            return node, node.outputs[0]

        node_prefix = 'FunctionNode'
        if type == DataType.BOOL:
            node = tree.nodes.new(node_prefix + 'InputBool')
            if value is not None:
                node.boolean = value
        elif type == DataType.INT:
            node = tree.nodes.new(node_prefix + 'InputInt')
            if value is not None:
                node.integer = value
        elif type == DataType.RGBA:
            node = tree.nodes.new(node_prefix + 'InputColor')
            if value is not None:
                node.color = value
        elif type == DataType.STRING:
            node = tree.nodes.new(node_prefix + 'InputString')
            if value is not None:
                node.string = value
        else:
            assert False, 'Unreachable, problem in type checker'
        return node, node.outputs[0]

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
        variables: dict[str, NodeSocket] = {}
        # Parse the input string into a sequence of tokens
        compiler = Compiler()
        success = compiler.compile(formula, file_loading.file_data.macros)
        if not success:
            return {'CANCELLED'}
        checked_program = compiler.checked_program
        for operation in checked_program:
            op_type = operation.op_type
            op_data = operation.data
            assert OpType.END_OF_STATEMENT.value == 7, 'Exhaustive handling of Operation types.'
            if op_type == OpType.PUSH_VALUE:
                stack.append(op_data)
            elif op_type == OpType.CREATE_VAR:
                assert isinstance(
                    op_data, str), 'Variable name should be a string.'
                socket = stack.pop()
                assert isinstance(
                    socket, NodeSocket), 'Bug in type checker, create var expects a node socket.'
                variables[op_data] = socket
                # root_nodes.append(socket.node)
            elif op_type == OpType.GET_VAR:
                assert isinstance(
                    op_data, str), 'Variable name should be a string.'
                stack.append(variables[op_data])
            elif op_type == OpType.GET_OUTPUT:
                assert isinstance(
                    op_data, int), 'Bug in type checker, index should be int.'
                index = op_data
                struct = stack[-1]
                assert isinstance(
                    struct, list), 'Bug in type checker, get_output only works on structs.'
                stack.append(struct[index])
            elif op_type == OpType.SWAP_2:
                a1 = stack.pop()
                a2 = stack.pop()
                stack += [a1, a2]
            elif op_type == OpType.CALL_FUNCTION:
                assert isinstance(
                    op_data, NodeFunction), 'Bug in type checker.'
                function: NodeFunction = op_data
                args = self.get_args(stack, len(function.input_sockets()))
                node = self.add_func(context, args, function)
                outputs = function.output_sockets()
                if len(outputs) == 1:
                    stack.append(node.outputs[outputs[0].index])
                elif len(outputs) > 1:
                    stack.append([node.outputs[socket.index]
                                  for socket in outputs])
                nodes.append(node)
            elif op_type == OpType.CREATE_INPUT:
                value = stack.pop()
                assert not isinstance(
                    value, NodeSocket), 'Only create Inputs for real values'
                node, socket = self.get_value_as_socket(value, op_data, tree)
                stack.append(socket)
                nodes.append(node)
            elif op_type == OpType.END_OF_STATEMENT:
                if stack == []:
                    continue
                element = stack.pop()
                if isinstance(element, NodeSocket):
                    root_nodes.append(element.node)
                stack = []
            else:
                print(f'Need implementation of {op_type}')
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
            res = compiler.compile(self.formula, file_loading.file_data.macros)
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
            res = compiler.compile(self.formula, file_loading.file_data.macros)
            self.errors = compiler.errors
            if res:
                self.report({'INFO'}, 'No errors detected')
            else:
                self.report({'WARNING'}, 'Compilation failed')
                print('Instructions: ', compiler.instructions)
                print('Checked Program: ', compiler.checked_program)

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
