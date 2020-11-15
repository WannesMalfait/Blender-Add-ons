import bpy
import bpy_extras
from math import *
import mathutils

bl_info = {
    "name": "Parametric",
    "location": "View3D > Add > Mesh > Add a parametric surface",
    "category": "Add Mesh",
    "description": "Add a parametric surface",
    "author": "Wannes Malfait",
    "version": (1, 0),
    "blender": (2, 90, 0),
}


class Parameters(bpy.types.PropertyGroup):
    """Parameters for the Parametric surface"""
    Unum : bpy.props.IntProperty(
        name="Unum",
        description="Number of u faces",
        default=20,
        min=1,)
    Vnum : bpy.props.IntProperty(
        name="Vnum",
        description="Number of v faces",
        default=4,
        min=1,)
    u_from : bpy.props.FloatProperty(
        name="u from",
        default = -pi,
    )
    v_from : bpy.props.FloatProperty(
        name="v from",
        default = -0.5,
    )
    u_to : bpy.props.FloatProperty(
        name="u to",
        default = pi,
    )
    v_to : bpy.props.FloatProperty(
        name="v to",
        default = 0.5,
    )
    help_a : bpy.props.StringProperty(
        name= "a",
        default = "2+v*cos(u/2)",
    )
    help_b : bpy.props.StringProperty(
        name= "b",
        default = "2",
    )
    x_func : bpy.props.StringProperty(
        name="X ",
        default = "a*cos(u)*b",
    )
    y_func : bpy.props.StringProperty(
        name="Y ",
        default = "a*sin(u)*b",
    )
    z_func : bpy.props.StringProperty(
        name="Z ",
        default = "v*sin(u/2)*b",
    )
    
    Subdivision : bpy.props.IntProperty(
        name="Subdivision",
        description="Subdivide the mesh (doesn't increase detail, 0 is no subdivision)",
        min=0,
        max=6,
        default=0,
        step = 1)
    Smooth_Shading : bpy.props.BoolProperty(
        name="Smooth shading",
        description="Set shading to smooth")
    Merge_Doubles : bpy.props.BoolProperty(
        name="Merge Doubles",
        description="Merge duplicated vertices")


class MESH_OT_add_parametric(bpy.types.Operator, bpy_extras.object_utils.AddObjectHelper):
    """Creates the Parametric surface"""
    bl_idname = "mesh.add_parametric"
    bl_label = "Add a Parametric Surface"
    bl_options = {'REGISTER', 'UNDO'}
    
    p: bpy.props.PointerProperty(type=Parameters)

    def execute(self, context):
        p = self.p
        # mesh arrays
        verts = []
        faces = []
        edges = []
        

        Unum = p.Unum
        Vnum = p.Vnum

        Uinc = (p.u_to-p.u_from)/Unum
        Vinc = (p.v_to-p.v_from)/Vnum
        
        # Compile the expressions before the loop starts
        a_code = compile(p.help_a, '<string>', 'eval')
        b_code = compile(p.help_b, '<string>', 'eval')
        x_code = compile(p.x_func, '<string>', 'eval')
        y_code = compile(p.y_func, '<string>', 'eval')
        z_code = compile(p.z_func, '<string>', 'eval')
        
        # fill verts array
        u = p.u_from
        for i in range(0, Unum + 1):
            v = p.v_from
            # Superformula
            for j in range(0, Vnum + 1):
                a = eval(a_code)
                b = eval(b_code)
                x = eval(x_code)
                y = eval(y_code)
                z = eval(z_code)
                vert = (x, y, z)
                verts.append(vert)
                # increment phi
                v = v + Vinc
            # increment theta
            u = u + Uinc
        # fill faces array
        count = 0
        for i in range(0, (Vnum + 1) * (Unum)):
            if count < Vnum:
                A = i
                B = i+1
                C = (i+(Vnum+1))+1
                D = (i+(Vnum+1))

                face = (A, B, C, D)
                faces.append(face)

                count = count + 1
            else:
                count = 0

        # create mesh and object
        mymesh = bpy.data.meshes.new("Parametric")

        # create mesh from python data
        mymesh.from_pydata(verts, edges, faces)
        mymesh.update(calc_edges=True)
        bpy_extras.object_utils.object_data_add(context, mymesh, operator=self)
        # test
        #go to editmode
        bpy.ops.object.editmode_toggle()


        # remove duplicate vertices
        if p.Merge_Doubles:
            bpy.ops.mesh.remove_doubles()

        # recalculate normals
        bpy.ops.mesh.normals_make_consistent(inside=False)
        #go back to objectmode
        bpy.ops.object.editmode_toggle()

        # Control the detail level
        if p.Subdivision != 0:
            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.object.modifiers["Subdivision"].levels = p.Subdivision
        if p.Smooth_Shading:
            mypolys = mymesh.polygons
            for p in mypolys:
                p.use_smooth = True

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        p = self.p
        col.label(text = "Help variables")
        col.prop(p, "help_a")
        col.prop(p, "help_b")
        col.label(text = "Functions")
        col.prop(p, "x_func")
        col.prop(p, "y_func")
        col.prop(p, "z_func")
        
        row = layout.row(align=True)
        row.label(text = "Domain u")
        row.prop(p, "u_from")
        row.prop(p, "u_to")
        row = layout.row(align=True)
        row.label(text = "Domain v")
        row.prop(p, "v_from")
        row.prop(p, "v_to")
        
        col = layout.column(align=True)
        # Control over the detail level
        col.label(text="Detail level:")
        col.prop(p, "Unum")
        col.prop(p, "Vnum")
        col.prop(p, "Subdivision")
        col.prop(p, "Smooth_Shading")
        col.prop(p, "Merge_Doubles")

def addMenu(self, context):
    self.layout.operator(MESH_OT_add_parametric.bl_idname,
                         text="Add Parametric",
                         icon="OUTLINER_OB_SURFACE")


classes = (
    Parameters,
    MESH_OT_add_parametric,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_MT_mesh_add.append(addMenu)
    


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    bpy.types.VIEW3D_MT_mesh_add.remove(addMenu)
    

if __name__ == "__main__":
    register()
