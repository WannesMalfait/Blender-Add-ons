# type: ignore
import bpy
import rna_keymap_ui

from . import (
    file_loading,
    main,
)



class MFMathFormula(bpy.types.AddonPreferences):
    bl_idname = __package__

    custom_implementations_folder: bpy.props.StringProperty(
        name="Custom implementations Folder",
        description="The folder where custom implementations should be loaded from",
        default=file_loading.custom_implementations_dir,
        subtype="DIR_PATH",
    )

    font_size: bpy.props.IntProperty(
        name="Font Size",
        description="Font size used for displaying text",
        default=15,
        min=8,
    )
    node_distance: bpy.props.IntProperty(
        name="Distance between nodes",
        description="The distance placed between the nodes from the formula",
        default=20,
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
        default=40,
        min=0,
    )
    show_colors: bpy.props.BoolProperty(
        name="Show colors for syntax highlighting",
        default=False,
    )
    background_color: bpy.props.FloatVectorProperty(
        name="Background Color",
        default=(0.1, 0.1, 0.1),
        subtype="COLOR",
    )
    background_alpha: bpy.props.FloatProperty(
        name="Background Alpha",
        default=0.9,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
    )
    python_color: bpy.props.FloatVectorProperty(
        # C586C0
        name="Python Color",
        default=(0.773, 0.525, 0.753),
        subtype="COLOR",
    )
    number_color: bpy.props.FloatVectorProperty(
        # B5CEA8
        name="Number Color",
        default=(0.71, 0.808, 0.659),
        subtype="COLOR",
    )
    string_color: bpy.props.FloatVectorProperty(
        # CE9178
        name="String Color",
        default=(0.808, 0.569, 0.471),
        subtype="COLOR",
    )
    identifier_color: bpy.props.FloatVectorProperty(
        # 9CDCFE
        name="Identifier Color",
        default=(0.612, 0.863, 0.996),
        subtype="COLOR",
    )
    default_color: bpy.props.FloatVectorProperty(
        name="Default Color",
        default=(1.0, 1.0, 1.0),
        subtype="COLOR",
    )
    keyword_color: bpy.props.FloatVectorProperty(
        # 569CD6
        name="Keyword Color",
        default=(0.337, 0.612, 0.839),
        subtype="COLOR",
    )
    type_color: bpy.props.FloatVectorProperty(
        # 4EC9B0
        name="Type Color",
        default=(0.306, 0.788, 0.69),
        subtype="COLOR",
    )
    function_color: bpy.props.FloatVectorProperty(
        # DCDCAA
        name="Function Color",
        default=(0.863, 0.863, 0.667),
        subtype="COLOR",
    )
    error_color: bpy.props.FloatVectorProperty(
        # F44747
        name="Error Color",
        default=(0.957, 0.278, 0.278),
        subtype="COLOR",
    )

    def draw(self: bpy.types.AddonPreferences, context: bpy.types.Context):
        layout = self.layout
        col = layout.column()
        col.prop(self, "font_size")
        col.prop(self, "node_distance")
        col.prop(self, "sibling_distance")
        col.prop(self, "subtree_distance")
        col.separator()
        col.prop(self, "custom_implementations_folder")
        col.label(
            text=f"{file_loading.file_data.num_funcs()} custom implementations are currently loaded."
        )
        col.operator(
            file_loading.MF_OT_load_custom_implementations.bl_idname, icon="FILE_PARENT"
        )
        props = col.operator(
            file_loading.MF_OT_load_custom_implementations.bl_idname,
            icon="FILE_REFRESH",
            text="Reload Custom Implementations",
        )
        props.force_update = True
        col.separator()
        col.prop(self, "show_colors")
        if self.show_colors:
            box = layout.box()
            box.label(text="Syntax Highlighting")
            box.prop(self, "background_color")
            box.prop(self, "background_alpha")
            box.prop(self, "python_color")
            box.prop(self, "number_color")
            box.prop(self, "string_color")
            box.prop(self, "identifier_color")
            box.prop(self, "default_color")
            box.prop(self, "keyword_color")
            box.prop(self, "type_color")
            box.prop(self, "function_color")
            box.prop(self, "error_color")
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
        description="Formula from which nodes are added",
        default="abs(sin(5*x))",
    )
    add_frame: bpy.props.BoolProperty(
        name="Add Frame",
        description="Put all the nodes in a frame",
        default=False,
    )


class MF_PT_add_panel(bpy.types.Panel, main.MFBase):
    bl_idname = "NODE_PT_mf_add_math_formula"
    bl_space_type = "NODE_EDITOR"
    bl_label = "Add Math Formula"
    bl_region_type = "UI"
    bl_category = "Math Formula"

    def draw(self, context: bpy.types.Context):
        # Helper variables
        layout = self.layout
        scene = context.scene
        props = scene.math_formula_add

        col = layout.column(align=True)
        col.label(text="Addon Preferences has more settings")
        col.prop(props, "formula")
        col.prop(props, "add_frame")
        col.separator()
        col.operator(main.MF_OT_math_formula_add.bl_idname)
        if context.active_node is not None:
            col.operator(main.MF_OT_arrange_from_root.bl_idname)
        else:
            col.label(text="--no active node--")


class MF_PT_file_panel(bpy.types.Panel, main.MFBase):
    bl_idname = "NODE_PT_mf_files"
    bl_space_type = "NODE_EDITOR"
    bl_label = "Change File Settings"
    bl_region_type = "UI"
    bl_category = "Math Formula"

    def draw(self, context: bpy.types.Context):
        # Helper variables
        layout = self.layout
        scene = context.scene
        props = scene.math_formula_add

        col = layout.column(align=True)
        col.label(
            text=f"{file_loading.file_data.num_funcs()} custom implementations are currently loaded."
        )
        col.operator(
            file_loading.MF_OT_load_custom_implementations.bl_idname, icon="FILE_PARENT"
        )
        props = col.operator(
            file_loading.MF_OT_load_custom_implementations.bl_idname,
            icon="FILE_REFRESH",
            text="Reload Custom Implementations",
        )
        props.force_update = True
        col.operator(
            file_loading.MF_OT_generate_node_info.bl_idname, icon="FILE_REFRESH"
        )


addon_keymaps = []
kmi_defs = [
    # kmi_defs entry: (identifier, key, action, CTRL, SHIFT, ALT, props)
    # props entry: (property name, property value)
    (
        main.MF_OT_arrange_from_root.bl_idname,
        "E",
        "PRESS",
        False,
        False,
        False,
        (("selected_only", False), ("invert_relations", False)),
    ),
    (
        main.MF_OT_arrange_from_root.bl_idname,
        "E",
        "PRESS",
        False,
        False,
        True,
        (("selected_only", True), ("invert_relations", False)),
    ),
    (
        main.MF_OT_arrange_from_root.bl_idname,
        "E",
        "PRESS",
        False,
        True,
        False,
        (("selected_only", False), ("invert_relations", True)),
    ),
    (
        main.MF_OT_arrange_from_root.bl_idname,
        "E",
        "PRESS",
        False,
        True,
        True,
        (("selected_only", True), ("invert_relations", True)),
    ),
    (
        main.MF_OT_select_from_root.bl_idname,
        "K",
        "PRESS",
        False,
        False,
        False,
        (("select_children", True), ("select_parents", True)),
    ),
    (
        main.MF_OT_select_from_root.bl_idname,
        "K",
        "PRESS",
        True,
        False,
        False,
        (("select_children", True), ("select_parents", False)),
    ),
    (
        main.MF_OT_select_from_root.bl_idname,
        "K",
        "PRESS",
        False,
        True,
        False,
        (("select_children", False), ("select_parents", True)),
    ),
    (
        main.MF_OT_type_formula_then_add_nodes.bl_idname,
        "F",
        "PRESS",
        False,
        False,
        True,
        None,
    ),
]

classes = (
    MFMathFormula,
    MF_Settings,
    MF_PT_add_panel,
    MF_PT_file_panel,
)


def register():
    file_loading.register()
    main.register()
    for cls in classes:
        bpy.utils.register_class(cls)

    # keymaps (Code from node wrangler)
    addon_keymaps.clear()
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
        for identifier, key, action, CTRL, SHIFT, ALT, props in kmi_defs:
            kmi = km.keymap_items.new(
                identifier, key, action, ctrl=CTRL, shift=SHIFT, alt=ALT
            )
            kmi.active = True
            if props:
                for prop, value in props:
                    setattr(kmi.properties, prop, value)
            addon_keymaps.append((km, kmi))

    bpy.types.Scene.math_formula_add = bpy.props.PointerProperty(type=MF_Settings)


def unregister():
    main.unregister()
    file_loading.unregister()
    for cls in classes:
        bpy.utils.unregister_class(cls)

    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    del bpy.types.Scene.math_formula_add


if __name__ == "__main__":
    register()
