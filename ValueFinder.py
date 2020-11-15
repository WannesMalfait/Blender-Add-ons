

import bpy
import bmesh  # For setting up the scene
from pathlib import Path  # Clearing the folder


bl_info = {
    "name": "Node Value Finder",
    "author": "Wannes Malfait",
    "version": (1, 0, 1),
    "location": "Node Editor Toolbar",
    "description": "Find out what values of each node do",
    "warning": "Be careful when using 'Clean Folder' ",
    "category": "Node",
    "blender": (2, 90, 0),  # Required so the add-on will actually load
}


def find_inputs_cb(self, context):

    # We can't get the inputs if there is no active node
    if context.active_node is None:
        return []

    node_active = context.active_node
    inputs = []
    for socket in node_active.inputs:
        if not socket.enabled or not socket.type == 'VALUE' or socket.is_linked:
            continue
        name = socket.label
        if not socket.label:
            name = socket.name
        inputs.append((name, name, name))
    return inputs


class VF_Settings(bpy.types.PropertyGroup):
    steps: bpy.props.IntProperty(
        name="Steps",
        description="Number of inbetween frames",
        default=40,
        min=2,
        soft_max=100,
    )
    input: bpy.props.EnumProperty(
        name="Input",
        description="Values are generated for this input",
        items=find_inputs_cb,
    )
    start_input: bpy.props.FloatProperty(
        name='Starting value',
        default=0,
    )
    end_input: bpy.props.FloatProperty(
        name='End value',
        default=1,
    )
    del_files: bpy.props.BoolProperty(
        name='Clean Folder',
        description='Remove files from the directory before creating new ones',
        default=True,
    )
    image_info: bpy.props.BoolProperty(
        name='Image Info',
        description='Add info about the value to the image name',
        default=True,
    )


class VF_OT_value_finder(bpy.types.Operator):
    """Render the scene for the different values"""
    bl_idname = "node.vf_value_finder"
    bl_label = "Value Finder"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR'

    def execute(self, context):
        node_active = context.active_node
        props = context.scene.value_finder
        rd = context.scene.render

        # "Clean" the folder, see:
        # https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
        if props.del_files:
            # WORKS??? I hope I didn't delete unwanted files :|
            # Added "_" to make sure that we don't go to the parent directory if no filename was given
            # Just the folder not the filename
            dir_path = Path(rd.filepath+"_").parent
            print(dir_path)
            [f.unlink() for f in dir_path.glob("*") if f.is_file()]

        input_socket = node_active.inputs[props.input]
        # Keep track of the originals
        original_value = input_socket.default_value
        original_path = rd.filepath
        for step in range(props.steps):

            # Steps is at least 2, so no zero division errors
            slope = (props.end_input-props.start_input)/(props.steps-1)
            value = props.start_input+slope*step
            input_socket.default_value = value
            if props.image_info:
                rd.filepath = f"{original_path}_({input_socket.name}_{step+1:n}={value:.2f})"
            else:
                rd.filepath = f"{original_path}{step+1:n}"
            bpy.ops.render.render(write_still=True)
        # Restore the originals
        rd.filepath = original_path
        input_socket.default_value = original_value
        self.report({'INFO'}, "Finished Rendering")
        return {'FINISHED'}


class VF_OT_cam_2d_setup(bpy.types.Operator):
    """Setup a scene for 2d textures"""
    bl_idname = "node.vf_cam_2d_setup"
    bl_label = "2D Setup"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR'

    def execute(self, context):
        scene = bpy.context.scene
        collections = bpy.data.collections
        active_mat = context.active_object.active_material

        # Keep things in our collection
        vf_collection = collections.get("vf_collection")
        if vf_collection is None:
            vf_collection = collections.new("vf_collection")
            scene.collection.children.link(vf_collection)

        # Create/get the plane
        plane = vf_collection.objects.get("vf_plane")
        if plane is None:
            mesh = bpy.data.meshes.new("vf_plane")
            bm = bmesh.new()
            bmesh.ops.create_grid(bm,
                                  x_segments=2,
                                  y_segments=2,
                                  size=1,
                                  calc_uvs=True)
            bm.to_mesh(mesh)
            bm.free()
            plane = bpy.data.objects.new("vf_plane", mesh)
            vf_collection.objects.link(plane)
        # Assign the material to the plane
        if plane.data.materials:
            # assign to 1st material slot
            plane.data.materials[0] = active_mat
        else:
            # no slots
            plane.data.materials.append(active_mat)

        sphere = vf_collection.objects.get("vf_sphere")
        if sphere is not None:
            sphere.hide_viewport = True
            sphere.hide_render = True
        plane.hide_viewport = False
        plane.hide_render = False

        camera = bpy.data.cameras.get("vf_camera")
        cam_obj = bpy.data.objects.get("vf_camera")
        if camera is None:
            camera = bpy.data.cameras.new("vf_camera")
            cam_obj = bpy.data.objects.new("vf_camera", camera)
            vf_collection.objects.link(cam_obj)

        # 2D Setup
        camera.ortho_scale = 2
        camera.type = 'ORTHO'
        cam_obj.location = (0, 0, 1)
        scene.camera = cam_obj
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        return {'FINISHED'}


class VF_OT_cam_3d_setup(bpy.types.Operator):
    """Setup a scene for 3d textures"""
    bl_idname = "node.vf_cam_3d_setup"
    bl_label = "3D Setup"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR'

    def execute(self, context):
        scene = bpy.context.scene
        collections = bpy.data.collections
        active_mat = context.active_object.active_material

        # Keep things in our collection
        vf_collection = collections.get("vf_collection")
        if vf_collection is None:
            vf_collection = collections.new("vf_collection")
            scene.collection.children.link(vf_collection)

        # Create/get the sphere
        sphere = vf_collection.objects.get("vf_sphere")
        if sphere is None:
            mesh = bpy.data.meshes.new("vf_sphere")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm,
                                      u_segments=32,
                                      v_segments=32,
                                      diameter=2,
                                      calc_uvs=True)
            bm.to_mesh(mesh)
            bm.free()
            sphere = bpy.data.objects.new("vf_sphere", mesh)
            vf_collection.objects.link(sphere)
        # Assign the material to the plane
        if sphere.data.materials:
            # assign to 1st material slot
            sphere.data.materials[0] = active_mat
        else:
            # no slots
            sphere.data.materials.append(active_mat)

        # Only show the sphere
        plane = vf_collection.objects.get("vf_plane")
        if plane is not None:
            plane.hide_viewport = True
            plane.hide_render = True
        sphere.hide_viewport = False
        sphere.hide_render = False

        camera = bpy.data.cameras.get("vf_camera")
        cam_obj = bpy.data.objects.get("vf_camera")
        if camera is None:
            camera = bpy.data.cameras.new("vf_camera")
            cam_obj = bpy.data.objects.new("vf_camera", camera)
            vf_collection.objects.link(cam_obj)

        # 3D Setup
        camera.type = 'PERSP'
        cam_obj.location = (0, 0, 6)
        scene.camera = cam_obj
        scene.render.resolution_x = 1024
        scene.render.resolution_y = 1024
        return {'FINISHED'}


class VF_OT_isolate_collection(bpy.types.Operator):
    """Isolate the View Finder collection for viewport/render """
    bl_idname = "node.vf_isolate_collection"
    bl_label = "Isolate Collection"

    @classmethod
    def poll(cls, context):
        space = context.space_data
        return space.type == 'NODE_EDITOR'

    def execute(self, context):
        collections = bpy.data.collections
        vf_collection = collections.get("vf_collection")
        for collection in collections:
            if collection == vf_collection:
                continue
            collection.hide_render = True
            collection.hide_viewport = True
        return{'FINISHED'}


class VF_PT_panel(bpy.types.Panel):
    bl_idname = "NODE_PT_vf_value_finder"
    bl_space_type = 'NODE_EDITOR'
    bl_label = "Value Finder"
    bl_region_type = "UI"
    bl_category = "Value Finder"

    def draw(self, context):

        # Helper variables
        layout = self.layout
        scene = context.scene
        props = scene.value_finder

        col = layout.column(align=True)
        # These should always be available
        col.operator(VF_OT_cam_2d_setup.bl_idname, icon='MESH_PLANE')
        col.operator(VF_OT_cam_3d_setup.bl_idname, icon='MESH_UVSPHERE')
        col.operator(VF_OT_isolate_collection.bl_idname)

        # Prevent some errors
        if context.active_node is None:
            col.label(text='No active node')
            return

        node = context.active_node

        if len(node.inputs) == 0:
            col.label(text='Active node has no input')
            return
        if len(node.outputs) == 0:
            col.label(text='Active node has no output')
            return

        # Start drawing
        col.separator()
        col.prop(props, 'input')
        if props.input is None or props.input == '':
            col.label(text='Select an input')
            return
        input_socket = node.inputs[props.input]
        col.prop(props, 'steps')
        col.prop(props, 'start_input')
        col.prop(props, 'end_input')
        col.separator()

        # For Convenience
        layout.use_property_split = True
        layout.use_property_decorate = False  # Not animatable
        col = layout.column(align=True)

        col.label(text="Output settings:")

        rd = scene.render
        col.prop(rd, 'filepath')
        col.prop(rd, 'engine')
        col.prop(rd, "resolution_x", text="Resolution X")
        col.prop(rd, "resolution_y", text="Y")
        col.prop(rd, "resolution_percentage", text="%")
        col.prop(rd, "film_transparent", text="Transparent Background")
        col.separator()
        col.label(text="Be careful with Clean Folder!")
        col.prop(props, 'del_files')
        col.prop(props, 'image_info')
        col.separator()

        col.operator(VF_OT_value_finder.bl_idname,
                     text="Generate Values", icon='OUTPUT')


classes = (
    VF_Settings,
    VF_OT_value_finder,
    VF_PT_panel,
    VF_OT_cam_2d_setup,
    VF_OT_cam_3d_setup,
    VF_OT_isolate_collection,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.value_finder = bpy.props.PointerProperty(type=VF_Settings)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.value_finder


if __name__ == "__main__":
    register()
