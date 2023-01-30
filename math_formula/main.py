import bpy
import traceback
from typing import cast
from bpy.types import Node
from . import file_loading
from .interpreter import Interpreter
from .positioning import TreePositioner
from .editor import Editor
from .compiler import Compiler


def mf_check(context: bpy.types.Context) -> bool:
    space = cast(bpy.types.SpaceNodeEditor, context.space_data)
    possible_trees = ('GeometryNodeTree', 'ShaderNodeTree')
    return space.type == 'NODE_EDITOR' and space.node_tree is not None and \
        space.tree_type in possible_trees


class MFBase:
    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return mf_check(context)


class MF_OT_select_from_root(bpy.types.Operator):
    """Select nodes linked to the active node """
    bl_idname = "node.mf_select_from_root"
    bl_label = "Select nodes from active"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context: bpy.types.Context) -> bool:
        return mf_check(context) and context.active_node is not None

    select_parents: bpy.props.BoolProperty(
        name="Select Parents",
        description="Select all the parents of this node recursively",
        default=False,
    )  # type: ignore
    select_children: bpy.props.BoolProperty(
        name="Select Children",
        description="Select all the children of this node recursively",
        default=False,
    )  # type: ignore

    def select_parents_of(self, node: bpy.types.Node, links: bpy.types.NodeLinks, visited: list[bpy.types.Node]) -> None:
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

    def select_children_of(self, node: bpy.types.Node, links: bpy.types.NodeLinks, visited: list[bpy.types.Node]) -> None:
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
        space = cast(bpy.types.SpaceNodeEditor, context.space_data)
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
    def poll(cls, context: bpy.types.Context) -> bool:
        return mf_check(context) and context.active_node is not None

    def execute(self, context: bpy.types.Context):
        space = cast(bpy.types.SpaceNodeEditor, context.space_data)
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
    )  # type: ignore

    @ staticmethod
    def store_mouse_cursor(context: bpy.types.Context, event: bpy.types.Event):
        space = cast(bpy.types.SpaceNodeEditor, context.space_data)
        tree = space.edit_tree

        # convert mouse position to the View2D for later node placement
        if context.region.type == 'WINDOW':
            # convert mouse position to the View2D for later node placement
            space.cursor_location_from_region(
                event.mouse_region_x, event.mouse_region_y)
        else:
            space.cursor_location = tree.view_center

    def execute(self, context: bpy.types.Context):
        space = cast(bpy.types.SpaceNodeEditor, context.space_data)
        # Safe because of poll function
        tree = space.edit_tree
        props = context.scene.math_formula_add  # type: ignore
        # The formula that we parse.
        formula: str = props.formula
        # Parse the input string into a sequence of operations
        compiler = Compiler(cast(str, space.tree_type), file_loading.file_data)
        success = compiler.compile(formula)
        if not success:
            return {'CANCELLED'}
        # Execute the compiled operations
        interpreter = Interpreter(tree)
        for operation in compiler.operations:
            interpreter.operation(operation)
        # The nodes that we added
        nodes: list[Node] = interpreter.nodes
        self.node_group_trees: list[bpy.types.NodeTree] = list(
            interpreter.node_group_trees.values())
        if props.add_frame and nodes != []:
            # Add all nodes in a frame
            frame = tree.nodes.new(type='NodeFrame')
            frame.label = formula
            for node in nodes:
                node.parent = frame
            frame.update()
        self.root_nodes = [[] for _ in range(len(self.node_group_trees) + 1)]
        for node in nodes:
            invalid = False
            for socket in node.outputs:
                if socket.is_linked:
                    invalid = True
                    break
            if invalid:
                continue
            else:
                self.root_nodes[0].append(node)
        for i, tree in enumerate(self.node_group_trees):
            for node in tree.nodes:
                invalid = False
                for socket in node.outputs:
                    if socket.is_linked:
                        invalid = True
                        break
                if invalid:
                    continue
                else:
                    self.root_nodes[i + 1].append(node)
        return {'FINISHED'}

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        if self.root_nodes[0][0].dimensions.x == 0:
            return {'RUNNING_MODAL'}
        space = cast(bpy.types.SpaceNodeEditor, context.space_data)
        links = space.edit_tree.links
        cursor_loc = space.cursor_location if self.use_mouse_location else (
            0, 0)
        node_positioner = TreePositioner(context)
        cursor_loc = node_positioner.place_nodes(
            self.root_nodes[0], links, cursor_loc=cast(tuple[int, int], cursor_loc))
        for i, tree in enumerate(self.node_group_trees):
            node_positioner = TreePositioner(context)
            node_positioner.place_nodes(
                self.root_nodes[i+1], tree.links, cursor_loc=(-100, 100))

        return {'FINISHED'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        self.store_mouse_cursor(context, event)
        self.execute(context)
        if self.root_nodes == [[]]:
            return {'FINISHED'}
        else:
            # Hacky way to force an update such that node dimensions are correct
            context.window_manager.modal_handler_add(self)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}


formula_history = []
formula_history_index = 0


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
                self._handle, 'WINDOW')  # type: ignore

    def modal(self, context: bpy.types.Context, event: bpy.types.Event):
        global formula_history_index
        editor_action = False
        context.area.tag_redraw()

        if event.value == 'RELEASE':
            self.middle_mouse = False
        # elif here since this was a key release and not a key press.
        elif event.type == 'RET':
            if event.ctrl:
                # Exit when they press control + enter
                compiler = Compiler(
                    cast(str, cast(bpy.types.SpaceNodeEditor, context.space_data).tree_type), file_loading.file_data)
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
                    self._handle, 'WINDOW')  # type: ignore
                context.scene.math_formula_add.formula = formula  # type: ignore
                formula_history.append(formula)
                formula_history_index += 1
                # Deselect all the nodes before adding new ones
                bpy.ops.node.select_all(action='DESELECT')
                try:
                    bpy.ops.node.mf_math_formula_add(  # type: ignore
                        'INVOKE_DEFAULT', use_mouse_location=True)
                except:
                    self.internal_error(remove_handle=False)
                return {'FINISHED'}
            else:
                # Just add a new line
                self.editor.new_line()
                editor_action = True

        # Cancel when they press Esc or Rmb
        elif event.type in ('ESC', 'RIGHTMOUSE'):
            bpy.types.SpaceNodeEditor.draw_handler_remove(  # type: ignore
                self._handle, 'WINDOW')
            return {'CANCELLED'}

        # Compile and check for errors
        elif event.alt and event.type == 'C':
            compiler = Compiler(cast(str, cast(bpy.types.SpaceNodeEditor, context.space_data).tree_type),
                                file_loading.file_data)
            try:
                res = compiler.compile(self.editor.get_text())
                print('\nCompiled program:\n', *compiler.operations, sep='\n')
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
            prefs.font_size += 1  # type: ignore
        elif event.type == 'WHEELDOWNMOUSE':
            prefs = context.preferences.addons['math_formula'].preferences
            prefs.font_size = max(8, prefs.font_size-1)  # type: ignore

        # FORMULA HISTORY NAVIGATION
        elif event.ctrl and event.type == 'UP_ARROW':
            self.editor.replace_text(
                formula_history[formula_history_index])
            formula_history_index = max(formula_history_index - 1, 0)
        elif event.ctrl and event.type == 'DOWN_ARROW':
            self.editor.replace_text(
                formula_history[formula_history_index])
            formula_history_index = min(
                formula_history_index + 1, len(formula_history) - 1)

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
        elif event.type == 'BACK_SPACE':
            self.editor.delete_before_cursor()
            # Prevent over sensitive keys
            editor_action = True
        elif event.type == 'DEL':
            self.editor.delete_after_cursor()
            editor_action = True
        elif event.ctrl and event.type == 'V':
            # Paste from clipboard
            self.editor.paste_after_cursor(context.window_manager.clipboard)
            editor_action = True
        elif event.unicode != "" and event.unicode.isprintable():
            # Only allow printable characters
            self.editor.add_char_after_cursor(event.unicode)
            editor_action = True

        # AUTOCOMPLETE
        elif event.type == 'TAB':
            self.editor.try_auto_complete(
                cast(bpy.types.SpaceNodeEditor, context.space_data).edit_tree.bl_idname)

        if editor_action:
            # Now editing this one instead of just looking through the history
            formula_history_index = len(formula_history) - 1

        return {'RUNNING_MODAL'}

    def invoke(self, context: bpy.types.Context, event: bpy.types.Event):
        self.editor = Editor((event.mouse_region_x, event.mouse_region_y))
        args = (self.editor, context)
        self._handle = bpy.types.SpaceNodeEditor.draw_handler_add(
            Editor.draw_callback_px, args, 'WINDOW', 'POST_PIXEL')  # type: ignore
        # Stores the location of the formula before dragging MMB
        self.old_editor_loc = self.editor.pos
        self.old_mouse_loc = (0, 0)
        self.middle_mouse = False
        if formula_history == []:
            # Add the last formula as history
            formula_history.append(
                context.scene.math_formula_add.formula)  # type: ignore
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
