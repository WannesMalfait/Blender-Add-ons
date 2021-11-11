from .nodes.base import DataType, NodeFunction, Value, ValueType
from .nodes import functions as function_nodes
from .nodes import geometry as geometry_nodes
from .nodes import shading as shader_nodes
import bpy
import blf
import traceback
from collections import deque
from . import file_loading
from .file_loading import fonts
from .positioning import TreePositioner
from .scanner import Scanner, Token, TokenType
from .parser import Compiler, Error, OpType, string_to_data_type
from bpy.types import Event, Node, NodeSocket


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
    def store_mouse_cursor(context: bpy.context, event: bpy.types.Event):
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
        node = tree.nodes.new(type=function.name())
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

    def execute(self, context: bpy.context):
        space = context.space_data
        # Safe because of poll function
        tree: bpy.types.NodeTree = space.edit_tree
        props = context.scene.math_formula_add
        # The formula that we parse. Should be in Reverse Polish Notation
        formula: str = props.formula
        stack: list[ValueType] = []
        # The nodes that we added
        nodes: list[Node] = []
        # Variables in the form of output sockets
        variables: dict[str, NodeSocket] = {}
        # Parse the input string into a sequence of tokens
        compiler = Compiler()
        success = compiler.compile(
            formula, file_loading.file_data.macros, tree.bl_idname)
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
        for node in nodes:
            invalid = False
            for socket in node.outputs:
                if socket.is_linked:
                    invalid = True
                    break
            if invalid:
                continue
            else:
                self.root_nodes.append(node)
        return {'FINISHED'}

    def modal(self, context: bpy.context, event: bpy.types.Event):
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

    def invoke(self, context: bpy.context, event: bpy.types.Event):
        self.store_mouse_cursor(context, event)
        self.execute(context)
        if self.root_nodes == []:
            return {'FINISHED'}
        else:
            # Hacky way to force an update such that node dimensions are correct
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}


class Editor():
    def __init__(self, pos: tuple[float, float]) -> None:
        self.pos = pos
        self.lines: list[str] = [""]
        self.line_tokens: list[list[Token]] = [[]]
        self.cursor_col: int = 0
        self.cursor_row: int = 0
        self.draw_cursor_col: int = 0
        self.scanner = Scanner("")
        self.errors: list[Error] = []
        self.suggestions: deque[str] = deque()

    def try_auto_complete(self, tree_type: str) -> None:
        # TODO: Make sugggestions bettter when prev token is a dot.
        # This requires actual parsing to be able to tell what we are 'dotting'.
        # Ideally something like 'tex_coord().' would suggest
        # the outputs of tex_coord() like generated, object...
        # while something like 'uv_sphere().' would suggest
        # all the functions that have a geometry as first input.
        # For something like 'sin().' it should give all the ones which
        # can have a float or something that float can convert to.
        token_under_cursor = None
        for token in self.line_tokens[self.cursor_row]:
            if token.start < self.draw_cursor_col <= token.start + len(token.lexeme):
                token_under_cursor = token
                break
        if token_under_cursor is not None:
            if len(self.suggestions) != 0:
                suggestion = self.suggestions.popleft()
                self.replace_token(token_under_cursor, suggestion)
                self.suggestions.append(suggestion)
                return

            for name in file_loading.file_data.macros.keys():
                if name.startswith(token.lexeme):
                    self.suggestions.append(name)

            for name in function_nodes.functions.keys():
                if name.startswith(token.lexeme):
                    self.suggestions.append(name)
            names = None
            if tree_type == 'GeometryNodeTree':
                names = geometry_nodes.functions.keys()
            else:
                names = shader_nodes.functions.keys()
            for name in names:
                if name.startswith(token.lexeme):
                    self.suggestions.append(name)
            if len(self.suggestions) == 0:
                return
            suggestion = self.suggestions.popleft()
            self.replace_token(token_under_cursor, suggestion)
            self.suggestions.append(suggestion)

    def replace_token(self, token: Token, text: str) -> None:
        start = token.start
        end = start + len(token.lexeme)
        line = self.lines[self.cursor_row]
        first = line[:start] + text
        self.draw_cursor_col = len(first)
        self.cursor_col = self.draw_cursor_col
        self.lines[self.cursor_row] = first + line[end:]
        self.rescan_line()

    def cursor_up(self) -> None:
        self.suggestions.clear()
        if self.cursor_row == 0:
            return
        else:
            self.cursor_row -= 1
            # Make sure that we don't draw outside of the line, but
            # at the same time keep track of where the actual cursor is
            # in case we jump to a longer line next.
            self.draw_cursor_col = min(
                len(self.lines[self.cursor_row]), self.cursor_col)

    def cursor_down(self) -> None:
        self.suggestions.clear()
        if self.cursor_row == len(self.lines) - 1:
            return
        else:
            self.cursor_row += 1
            # Make sure that we don't draw outside of the line, but
            # at the same time keep track of where the actual cursor is
            # in case we jump to a longer line next.
            self.draw_cursor_col = min(
                len(self.lines[self.cursor_row]), self.cursor_col)

    def cursor_left(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == 0:
            if self.cursor_row != 0:
                self.cursor_row -= 1
                self.draw_cursor_col = len(self.lines[self.cursor_row])
        else:
            self.draw_cursor_col -= 1
        self.cursor_col = self.draw_cursor_col

    def cursor_right(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == len(self.lines[self.cursor_row]):
            if self.cursor_row != len(self.lines) - 1:
                self.cursor_row += 1
                self.draw_cursor_col = 0
        else:
            self.draw_cursor_col += 1
        self.cursor_col = self.draw_cursor_col

    def cursor_home(self) -> None:
        self.suggestions.clear()
        self.draw_cursor_col = 0
        self.cursor_col = self.draw_cursor_col

    def cursor_end(self) -> None:
        self.suggestions.clear()
        self.draw_cursor_col = len(self.lines[self.cursor_row])
        self.cursor_col = self.draw_cursor_col

    def delete_before_cursor(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == 0:
            if self.cursor_row == 0:
                self.cursor_col = 0
                return
            # Merge this line with previous one.
            self.draw_cursor_col = len(self.lines[self.cursor_row - 1])
            self.lines[self.cursor_row-1] += self.lines[self.cursor_row]
            self.cursor_row -= 1
            self.rescan_line()
            self.cursor_col = self.draw_cursor_col
            self.lines.pop(self.cursor_row+1)
            self.line_tokens.pop(self.cursor_row+1)
            return
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.draw_cursor_col -
                                           1] + line[self.draw_cursor_col:]
        self.draw_cursor_col -= 1
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def delete_after_cursor(self) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        self.cursor_col = self.draw_cursor_col
        if self.draw_cursor_col == len(line):
            if self.cursor_row == len(self.lines) - 1:
                return
            # Merge this next line with this one.
            self.lines[self.cursor_row] += self.lines[self.cursor_row + 1]
            self.rescan_line()
            self.lines.pop(self.cursor_row+1)
            self.line_tokens.pop(self.cursor_row+1)
            return
        self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
            line[self.draw_cursor_col + 1:]
        self.rescan_line()

    def paste_after_cursor(self, text: str) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        if (index := text.find('\n')) != -1:
            self.lines[self.cursor_row] = line[:self.draw_cursor_col] + text[:index]
            self.rescan_line()
            line = line[self.draw_cursor_col:]
            text = text[index+1:]
            self.draw_cursor_col = len(self.lines[self.cursor_row])
            self.new_line()
            while True:
                if text == "":
                    break
                if (index := text.find('\n')) != -1:
                    self.lines[self.cursor_row] = text[:index]
                    self.rescan_line()
                    text = text[index+1:]
                    self.draw_cursor_col = len(self.lines[self.cursor_row])
                    self.new_line()
                else:
                    self.lines[self.cursor_row] = text + line
                    self.draw_cursor_col = 0
                    self.rescan_line()
                    break
        else:
            self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
                text + line[self.draw_cursor_col:]
            self.rescan_line()
        self.draw_cursor_col += len(text)
        self.cursor_col = self.draw_cursor_col

    def add_char_after_cursor(self, char: str) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
            char + line[self.draw_cursor_col:]
        self.draw_cursor_col += 1
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def new_line(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col != len(self.lines[self.cursor_row]):
            line = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = line[:self.draw_cursor_col]
            self.rescan_line()
            self.cursor_row += 1
            self.lines.insert(self.cursor_row, line[self.draw_cursor_col:])
            self.line_tokens.insert(self.cursor_row, [])
            self.rescan_line()
            self.cursor_col = 0
            self.draw_cursor_col = 0
            return
        self.cursor_row += 1
        self.lines.insert(self.cursor_row, "")
        self.line_tokens.insert(self.cursor_row, [])
        self.cursor_col = 0
        self.draw_cursor_col = 0

    def rescan_line(self) -> None:
        line = self.cursor_row
        self.scanner.reset(self.lines[line])
        # Expects a 1-based index
        self.scanner.line = line + 1
        self.line_tokens[line] = []
        while(token := self.scanner.scan_token()).token_type != TokenType.EOL:
            self.line_tokens[line].append(token)

    def get_text(self) -> str:
        return '\n'.join(self.lines)

    def draw_callback_px(self, context: bpy.context):
        prefs = context.preferences.addons['math_formula'].preferences
        font_id = fonts['regular']
        font_size = prefs.font_size
        font_dpi = 72
        blf.size(font_id, font_size, font_dpi)

        char_width = blf.dimensions(font_id, 'H')[0]
        char_height = blf.dimensions(font_id, 'Hq')[1]*1.3
        # Set the initial positions of the text
        posx = self.pos[0]
        posy = self.pos[1]
        posz = 0

        # Get the dimensions so that we know where to place the next stuff
        width = blf.dimensions(font_id, "Formula: ")[0]
        # Color for the non-user text.
        blf.color(font_id, 0.4, 0.5, 0.1, 1.0)
        blf.position(font_id, posx, posy+char_height, posz)
        blf.draw(
            font_id, f"(Press ENTER to confirm, ESC to cancel)    (Line:{self.cursor_row+1} Col:{self.draw_cursor_col+1})")

        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, posx, posy, posz)
        blf.draw(font_id, "Formula: ")
        for line_num, tokens in enumerate(self.line_tokens):
            line = self.lines[line_num]
            prev = 0
            line_posx = posx+width
            line_posy = posy - char_height*line_num
            for i, token in enumerate(tokens):
                blf.position(font_id, line_posx, line_posy, posz)
                text = token.lexeme
                start = token.start
                # Draw white space
                white_space = line[prev:start]
                for char in white_space:
                    blf.position(font_id, line_posx, line_posy, posz)
                    blf.draw(font_id, char)
                    line_posx += char_width
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
                blf.size(token_font_style, font_size, font_dpi)

                # Draw manually to ensure equal spacing and no kerning.
                for char in text:
                    blf.position(token_font_style, line_posx, line_posy, posz)
                    blf.draw(token_font_style, char)
                    line_posx += char_width
                prev = start + len(text)
            # Errors
            color(font_id, prefs.error_color)
            error_base_y = posy-char_height*(len(self.lines) + 1)
            for n, error in enumerate(self.errors):
                blf.position(font_id, posx+width,
                             error_base_y - n*char_height, posz)
                blf.draw(font_id, error.message)
                macro_token = error.token
                while macro_token.expanded_from is not None:
                    macro_token = macro_token.expanded_from
                error_col = macro_token.col - 1
                error_row = macro_token.line - 1
                blf.position(font_id, posx+width+char_width *
                             error_col, posy-char_height*error_row - char_height*0.75, posz)
                blf.draw(font_id, '^'*len(error.token.lexeme))
        # Draw cursor
        blf.color(font_id, 0.1, 0.4, 0.7, 1.0)
        blf.position(font_id, posx+width+self.draw_cursor_col*char_width-char_width/2,
                     posy-char_height*self.cursor_row, posz)
        blf.draw(font_id, '|')


def color(font_id, color):
    blf.color(font_id, color[0], color[1], color[2], 1.0)


class MF_OT_type_formula_then_add_nodes(bpy.types.Operator, MFBase):
    """Type the formula then add the nodes"""
    bl_idname = "node.mf_type_formula_then_add_nodes"
    bl_label = "Type math formula then add nodes"
    bl_options = {'REGISTER'}

    def internal_error(self, remove_handle: bool = True) -> None:
        self.report(
            {'ERROR'}, 'Internal error, please report as a bug (see console)')
        print('\n\nERROR REPORT:\n')
        traceback.print_exc()
        print('Error triggered by following formula:')
        print(self.editor.get_text())
        print(
            'Please report this at https://github.com/WannesMalfait/Blender-Add-ons/issues')
        if remove_handle:
            bpy.types.SpaceNodeEditor.draw_handler_remove(
                self._handle, 'WINDOW')

    def modal(self, context: bpy.context, event: bpy.types.Event):
        context.area.tag_redraw()
        if event.type == 'RET':
            if event.shift:
                # Ensure we go here if shift
                if (not self.lock or event.is_repeat):
                    self.editor.new_line()
                    self.lock = True
            else:
                # Exit when they press enter
                compiler = Compiler()
                formula = self.editor.get_text()
                try:
                    res = compiler.compile(
                        formula, file_loading.file_data.macros, context.space_data.edit_tree.bl_idname)
                except:
                    self.internal_error()
                    return {'CANCELLED'}
                if not res:
                    self.editor.errors = compiler.errors
                    self.report(
                        {'WARNING'}, 'Compile errors, could not create node tree')
                    return {'RUNNING_MODAL'}
                bpy.types.SpaceNodeEditor.draw_handler_remove(
                    self._handle, 'WINDOW')
                context.scene.math_formula_add.formula = formula
                # Deselect all the nodes before adding new ones
                bpy.ops.node.select_all(action='DESELECT')
                try:
                    bpy.ops.node.mf_math_formula_add(
                        'INVOKE_DEFAULT', use_mouse_location=True)
                except:
                    self.internal_error(remove_handle=False)
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
            try:
                res = compiler.compile(
                    self.editor.get_text(), file_loading.file_data.macros, context.space_data.edit_tree.bl_idname)
            except:
                self.internal_error()
                return {'CANCELLED'}
            self.editor.errors = compiler.errors
            if res:
                self.report({'INFO'}, 'No errors detected')
            else:
                self.report({'WARNING'}, 'Compilation failed')

        # NAVIGATION
        elif event.type == 'MIDDLEMOUSE':
            self.old_mouse_loc = (event.mouse_region_x, event.mouse_region_y)
            self.old_editor_loc = self.editor.pos
            self.middle_mouse = True
        elif event.type == 'MOUSEMOVE' and self.middle_mouse:
            self.editor.pos = (
                self.old_editor_loc[0] +
                event.mouse_region_x - self.old_mouse_loc[0],
                self.old_editor_loc[1] +
                event.mouse_region_y - self.old_mouse_loc[1])
        elif event.type == 'WHEELUPMOUSE':
            prefs = context.preferences.addons['math_formula'].preferences
            prefs.font_size += 1
        elif event.type == 'WHEELDOWNMOUSE':
            prefs = context.preferences.addons['math_formula'].preferences
            prefs.font_size = max(8, prefs.font_size-1)

        # CURSOR NAVIGATION
        elif event.type == 'LEFT_ARROW':
            self.editor.cursor_left()
        elif event.type == 'RIGHT_ARROW':
            self.editor.cursor_right()
        elif event.type == 'HOME':
            self.editor.cursor_home()
        elif event.type == 'END':
            self.editor.cursor_end()
        elif event.type == 'UP_ARROW':
            self.editor.cursor_up()
        elif event.type == 'DOWN_ARROW':
            self.editor.cursor_down()

        # INSERTION + DELETING
        elif (not self.lock or event.is_repeat) and event.type == 'BACK_SPACE':
            self.editor.delete_before_cursor()
            # Prevent over sensitive keys
            self.lock = True
        elif (not self.lock or event.is_repeat) and event.type == 'DEL':
            self.editor.delete_after_cursor()
            self.lock = True
        elif not self.lock and event.ctrl and event.type == 'V':
            # Paste from clipboard
            self.editor.paste_after_cursor(context.window_manager.clipboard)
            self.lock = True
        elif event.unicode != "" and event.unicode.isprintable():
            # Only allow printable characters
            self.editor.add_char_after_cursor(event.unicode)

        # AUTOCOMPLETE
        elif not self.lock and event.type == 'TAB':
            self.editor.try_auto_complete(
                context.space_data.edit_tree.bl_idname)
            self.lock = True
        return {'RUNNING_MODAL'}

    def invoke(self, context: bpy.context, event: bpy.types.Event):
        self.editor = Editor((event.mouse_region_x, event.mouse_region_y))
        args = (self.editor, context)
        self._handle = bpy.types.SpaceNodeEditor.draw_handler_add(
            Editor.draw_callback_px, args, 'WINDOW', 'POST_PIXEL')
        # Stores the location of the formula before dragging MMB
        self.old_editor_loc = self.editor.pos
        self.old_mouse_loc = (0, 0)
        self.lock = False
        self.middle_mouse = False
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
