import bpy
import bpy_extras
import math
bl_info = {
    "name": "Times Table",
    "location": "View3D > Add > Mesh",
    "category": "Add Mesh",
    "description": "Add a cool mathematical design",
    "author": "Wannes Malfait",
    "version": (1, 0),
    "blender": (2, 82, 0),
}


class Params(bpy.types.PropertyGroup):
    base: bpy.props.IntProperty(
        name="Base",
        description="Number of points around the circle",
        default=10,
        min=2,
        soft_max=256,
    )
    multiplier: bpy.props.IntProperty(
        name="Multiplier",
        description="Step by which points get connected",
        default=2,
        min=2,
        soft_max=128,
    )
    radius: bpy.props.FloatProperty(
        name="Radius",
        default=1,
        min=0.0001,
        soft_max=10,
    )
    skin_modifier: bpy.props.BoolProperty(
        name="Add Skin Modifier",
        default=False,
    )
    mean_radius: bpy.props.FloatVectorProperty(
        name="Mean Radius",
        default=(0.1, 0.1, 0.0),
    )


class MESH_OT_times_table(bpy.types.Operator, bpy_extras.object_utils.AddObjectHelper):
    """ Adds a mathematical times table as a mesh """
    bl_idname = "mesh.add_times_table"
    bl_label = "Times Table"
    bl_options = {'REGISTER', 'UNDO'}

    p: bpy.props.PointerProperty(type=Params)

    def execute(self, context):
        # Intialize the verts lists
        verts = [(0, 0, 0) for i in range(self.p.base)]
        edges = []
        for t in range(self.p.base):
            angle = t/self.p.base*math.pi*2
            verts[t] = (self.p.radius*math.cos(angle),
                        self.p.radius*math.sin(angle), 0)
            result = t*self.p.multiplier % self.p.base
            if(result != t):
                edges.append((t, result))

        mymesh = bpy.data.meshes.new("Times Table")

        # create mesh from python data
        mymesh.from_pydata(verts, edges, [])
        bpy_extras.object_utils.object_data_add(context, mymesh, operator=self)

        if self.p.skin_modifier:
            bpy.ops.object.modifier_add(type='SKIN')
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.object.skin_root_mark()
            bpy.ops.transform.skin_resize(value=self.p.mean_radius)
            bpy.ops.object.editmode_toggle()
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        # Control over the parameters
        col.label(text="Times Table")
        col.prop(self.p, "radius")
        col.prop(self.p, "base")
        col.prop(self.p, "multiplier")
        col.prop(self.p, "skin_modifier")
        if self.p.skin_modifier:
            col.prop(self.p, "mean_radius")


def draw_add_menu(self, context):
    self.layout.operator(MESH_OT_times_table.bl_idname,
                         text=MESH_OT_times_table.bl_label)


classes = [
    # Order is important here
    Params,
    MESH_OT_times_table,
]


def register():
    bpy.types.VIEW3D_MT_mesh_add.append(draw_add_menu)
    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    bpy.types.VIEW3D_MT_mesh_add.remove(draw_add_menu)
    for c in classes:
        bpy.utils.unregister_class(c)
