import bpy
import traceback
from bpy.types import Node, NodeSocket
from math_formula.backends.main import ValueType, OpType
from math_formula.positioning import TreePositioner
from math_formula.editor import Editor
from math_formula.compiler import Compiler
from math_formula.backends.geometry_nodes import GeometryNodesBackEnd


def mf_check(context: bpy.context) -> bool:
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

    def get_args(self, stack: list, num_args: int) -> list[ValueType]:
        args = stack[-num_args:]
        stack[:] = stack[:-num_args]
        return args

    # @staticmethod
    # def add_func(context: bpy.context, args: list[ValueType], function: NodeFunction):
    #     tree: bpy.types.NodeTree = context.space_data.edit_tree
    #     node = tree.nodes.new(type=function.name())
    #     for name, value in function.props():
    #         setattr(node, name, value)
    #     for i, socket in enumerate(function.input_sockets()):
    #         arg = args[i]
    #         if isinstance(arg, bpy.types.NodeSocket):
    #             tree.links.new(arg, node.inputs[socket.index])
    #         elif not arg is None:
    #             node.inputs[socket.index].default_value = arg
    #     return node

    @staticmethod
    def add_builtin(context: bpy.context, name: str, props: list[tuple], args: list[ValueType], used_inputs: list[int]) -> bpy.types.Node:
        tree: bpy.types.NodeTree = context.space_data.edit_tree
        node = tree.nodes.new(type=name)
        for name, value in props:
            setattr(node, name, value)
        for i, input_index in enumerate(used_inputs):
            arg = args[i]
            if isinstance(arg, bpy.types.NodeSocket):
                tree.links.new(arg, node.inputs[input_index])
            elif not arg is None:
                node.inputs[input_index].default_value = arg
        return node

    # @staticmethod
    # def get_value_as_socket(value: ValueType, type: DataType, tree: bpy.types.NodeTree) -> tuple[bpy.types.Node, NodeSocket]:
    #     node = None
    #     node_prefix = 'ShaderNode'
    #     if type == DataType.UNKNOWN or type == DataType.DEFAULT or type == DataType.FLOAT:
    #         node = tree.nodes.new(node_prefix + 'Value')
    #         if value is not None:
    #             node.outputs[0].default_value = value
    #         return node, node.outputs[0]

    #     node_prefix = 'FunctionNode'
    #     if type == DataType.BOOL:
    #         node = tree.nodes.new(node_prefix + 'InputBool')
    #         if value is not None:
    #             node.boolean = value
    #     elif type == DataType.INT:
    #         node = tree.nodes.new(node_prefix + 'InputInt')
    #         if value is not None:
    #             node.integer = value
    #     elif type == DataType.RGBA:
    #         node = tree.nodes.new(node_prefix + 'InputColor')
    #         if value is not None:
    #             node.color = value
    #     elif type == DataType.STRING:
    #         node = tree.nodes.new(node_prefix + 'InputString')
    #         if value is not None:
    #             node.string = value
    #     else:
    #         assert False, 'Unreachable, problem in type checker'
    #     return node, node.outputs[0]

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
        # TODO: Change this to match the tree type
        compiler = Compiler(GeometryNodesBackEnd())
        success = compiler.compile(formula)
        if not success:
            return {'CANCELLED'}
        for operation in compiler.operations:
            op_type = operation.op_type
            op_data = operation.data
            assert OpType.END_OF_STATEMENT.value == 7, 'Exhaustive handling of Operation types.'
            if op_type == OpType.PUSH_VALUE:
                stack.append(op_data)
            elif op_type == OpType.CREATE_VAR:
                raise NotImplementedError
                assert isinstance(
                    op_data, str), 'Variable name should be a string.'
                socket = stack.pop()
                assert isinstance(
                    socket, NodeSocket), 'Bug in type checker, create var expects a node socket.'
                variables[op_data] = socket
                # root_nodes.append(socket.node)
            elif op_type == OpType.GET_VAR:
                raise NotImplementedError
                assert isinstance(
                    op_data, str), 'Variable name should be a string.'
                stack.append(variables[op_data])
            elif op_type == OpType.GET_OUTPUT:
                raise NotImplementedError
                assert isinstance(
                    op_data, int), 'Bug in type checker, index should be int.'
                index = op_data
                struct = stack[-1]
                assert isinstance(
                    struct, list), 'Bug in type checker, get_output only works on structs.'
                stack.append(struct[index])
            elif op_type == OpType.CALL_FUNCTION:
                raise NotImplementedError
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
            elif op_type == OpType.CALL_BUILTIN:
                assert isinstance(op_data, tuple), 'Bug in compiler.'
                name, node_props = op_data
                used_indices = stack.pop()
                assert isinstance(
                    used_indices, tuple), 'Used inputs and outputs should be on top of the stack'
                used_inputs, used_outputs = used_indices
                args = self.get_args(stack, len(used_inputs))
                node = self.add_builtin(
                    context, name, node_props, args, used_inputs)
                if len(used_outputs) == 1:
                    stack.append(node.outputs[used_outputs[0]])
                elif len(used_outputs) > 1:
                    stack.append([node.outputs[i] for i in used_outputs])
                nodes.append(node)
            elif op_type == OpType.CALL_NODEGROUP:
                raise NotImplementedError
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
                # Exit when they press shift + enter
                # TODO: Change this based on tree type
                compiler = Compiler(GeometryNodesBackEnd())
                formula = self.editor.get_text()
                try:
                    res = compiler.compile(formula)
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
            else:
                # Just add a new line
                if (not self.lock or event.is_repeat):
                    self.editor.new_line()
                    self.lock = True

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
            # TODO: Change this based on tree type
            compiler = Compiler(GeometryNodesBackEnd())
            try:
                res = compiler.compile(self.editor.get_text())
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
        # elif not self.lock and event.type == 'TAB':
        #     self.editor.try_auto_complete(
        #         context.space_data.edit_tree.bl_idname)
        #     self.lock = True
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
