from . import main
import bpy
import rna_keymap_ui

bl_info = {
    "name": "Node Math Formula",
    "author": "Wannes Malfait",
    "version": (0, 5, 0),
    "location": "Node Editor Toolbar",
    "description": "Quickly add math nodes by typing in a formula",
    "category": "Node",
    "blender": (2, 93, 0),  # Required so the add-on will actually load
}


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
    sibling_distance: bpy.props.IntProperty(
        name="Distance between siblings",
        description="The distance between nodes which connect to the same node",
        default=20,
        min=0,
    )
    subtree_distance: bpy.props.IntProperty(
        name="Distance between subtrees",
        description="The distance between two subtrees",
        default=50,
        min=0,
    )
    show_colors: bpy.props.BoolProperty(
        name="Show colors for syntax highlighting",
        default=False,
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
        col.prop(self, 'sibling_distance')
        col.prop(self, 'subtree_distance')
        col.separator()
        col.prop(self, 'show_colors')
        if self.show_colors:
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
        description="The name of the attribute used to fill in missing arguments"
    )
    add_frame: bpy.props.BoolProperty(
        name="Add Frame",
        description='Put all the nodes in a frame',
        default=False,
    )


class MF_PT_panel(bpy.types.Panel, main.MFBase):
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
        col.prop(props, 'add_frame')
        col.separator()
        if context.space_data.tree_type == 'GeometryNodeTree':
            col.prop(props, 'temp_attr_name')
            col.prop(props, 'no_arg')
            col.operator(main.MF_OT_attribute_math_formula_add.bl_idname)
        col.operator(main.MF_OT_math_formula_add.bl_idname)
        if context.active_node is not None:
            col.operator(main.MF_OT_arrange_from_root.bl_idname)
        else:
            col.label(text="No active node")


addon_keymaps = []
kmi_defs = [
    # kmi_defs entry: (identifier, key, action, CTRL, SHIFT, ALT, props)
    # props entry: (property name, property value)
    (main.MF_OT_arrange_from_root.bl_idname,
     'E', 'PRESS', False, False, True, None),
    (main.MF_OT_type_formula_then_add_nodes.bl_idname,
     'F', 'PRESS', False, True, False, (('use_attributes', True),)),
    (main.MF_OT_type_formula_then_add_nodes.bl_idname,
     'F', 'PRESS', False, False, True, (('use_attributes', False),))
]

classes = (
    MFMathFormula,
    MF_Settings,
    MF_PT_panel,
)


def register():
    main.register()
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
    main.unregister()
    for cls in classes:
        bpy.utils.unregister_class(cls)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    del bpy.types.Scene.math_formula_add


if __name__ == "__main__":
    register()
