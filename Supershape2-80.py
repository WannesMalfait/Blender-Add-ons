import bpy
import bpy_extras
import math
import mathutils

bl_info = {
    "name": "Supershape",
    "location": "View3D > Add > Mesh > Add Supershape",
    "category": "Add Mesh",
    "description": "Add a supershape!",
    "author": "Wannes Malfait",
    "wiki_url": "https://en.wikipedia.org/wiki/Superformula",
    "version": (1, 0),
    "blender": (2, 80, 0),
}


class Params(bpy.types.PropertyGroup):
    """Parameters for the Supershape"""
    M : bpy.props.FloatProperty(
        name="M",
        description="Increases the amount of 'blobs'",
        default=6.00)
    A : bpy.props.FloatProperty(
        name="A",
        description="Influences the size (smaller values = bigger)",
        default=1.00,
        min=0.001)
    B : bpy.props.FloatProperty(
        name="B",
        description="Influences the size (smaller values = bigger)",
        default=1.00,
        min=0.001)
    n1: bpy.props.FloatProperty(
        name="n1",
        description="Exaggerates the 'blobs'",
        default=0.23)
    n2: bpy.props.FloatProperty(
        name="n2",
        description="Works similarly to n1 (has a more vertical effect)",
        default=2.66)
    n3: bpy.props.FloatProperty(
        name="n3",
        description="Works similarly to n1 (has a more horizontal effect)",
        default=1.49)
    Detail:  bpy.props.IntProperty(
        name="Detail",
        description="Amount of detail",
        default=40,
        min=1,
        max=400)
    Subdivision : bpy.props.IntProperty(
        name="Subdivision",
        description="Subdivide the mesh (doesn't increase detail, 0 is no subdivision)",
        min=0,
        max=6,
        default=0)
    Smooth_Shading : bpy.props.BoolProperty(
        name="Smooth shading",
        description="Set shading to smooth")


class MESH_OT_addSupershape(bpy.types.Operator, bpy_extras.object_utils.AddObjectHelper):
    """Creates the Supershape"""
    bl_idname = "mesh.add_supershape"
    bl_label = "Add Supershape"
    bl_options = {'REGISTER', 'UNDO'}

    p : bpy.props.PointerProperty(name="Parameters", type=Params)

    def execute(self, context):
        # mesh arrays
        verts = []
        faces = []
        edges = []

        # 3D supershape parameters
        m = self.p.M
        a = -self.p.A
        b = self.p.B
        if self.p.n1 != 0:
            n1 = self.p.n1
        else:
            n1 = 0.1
        n2 = self.p.n2
        n3 = self.p.n3

        scale = 1

        Unum = self.p.Detail
        Vnum = self.p.Detail

        Uinc = math.pi / (Unum/2)
        Vinc = (math.pi/2)/(Vnum/2)

        # fill verts array
        theta = -math.pi
        for i in range(0, Unum + 1):
            phi = -math.pi/2
            # Superformula
            r1 = ((abs(math.cos(m*theta/4)/a))**n2 +
                  (abs(math.sin(m*theta/4)/b))**n3)**(-1/n1)
            for j in range(0, Vnum + 1):
                r2 = ((abs(math.cos(m*phi/4)/a))**n2 +
                      (abs(math.sin(m*phi/4)/b))**n3)**(-1/n1)
                x = scale * (r1 * math.cos(theta) * r2 * math.cos(phi))
                y = scale * (r1 * math.sin(theta) * r2 * math.cos(phi))
                z = scale * (r2 * math.sin(phi))

                vert = (x, y, z)
                verts.append(vert)
                # increment phi
                phi = phi + Vinc
            # increment theta
            theta = theta + Uinc
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
        mymesh = bpy.data.meshes.new("Supershape")
        myobject = bpy.data.objects.new("Supershape", mymesh)

        # set mesh location
        myobject.location = bpy.context.scene.cursor.location
        #bpy.context.scene.objects.link(myobject)

        # create mesh from python data
        mymesh.from_pydata(verts, edges, faces)
        mymesh.update(calc_edges=True)
        bpy_extras.object_utils.object_data_add(context, mymesh, operator=self)
        #go to editmode
        bpy.ops.object.editmode_toggle()


        # remove duplicate vertices
        bpy.ops.mesh.remove_doubles()

        # recalculate normals
        bpy.ops.mesh.normals_make_consistent(inside=False)
        #go back to objectmode
        bpy.ops.object.editmode_toggle()

        # Control the detail level
        if self.p.Subdivision != 0:
            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.object.modifiers["Subdivision"].levels = self.p.Subdivision
        if self.p.Smooth_Shading:
            mypolys = mymesh.polygons
            for p in mypolys:
                p.use_smooth = True

        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        # Control over the detail level
        col.label(text="Detail level:")
        col.prop(self.p, "Detail")
        col.prop(self.p, "Subdivision")
        col.prop(self.p, "Smooth_Shading")
        # Parameters
        col.label(text="Parameters:")
        col.prop(self.p, "M")
        col.prop(self.p, "A")
        col.prop(self.p, "B")
        col.prop(self.p, "n1")
        col.prop(self.p, "n2")
        col.prop(self.p, "n3")


def addMenu(self, context):
    self.layout.operator(MESH_OT_addSupershape.bl_idname,
                         text="Add Supershape")


def register():
    bpy.utils.register_class(Params)
    bpy.utils.register_class(MESH_OT_addSupershape)
    bpy.types.VIEW3D_MT_mesh_add.append(addMenu)


def unregister():
    bpy.utils.unregister_class(Params)
    bpy.utils.unregister_class(MESH_OT_addSupershape)
    bpy.types.VIEW3D_MT_mesh_add.remove(addMenu)


if __name__ == "__main__":
    register()
