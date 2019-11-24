import bpy
import bpy_extras
import math
import mathutils

bl_info = {
    "name": "L-System",
    "location": "View3D > Add > Mesh > Add Fractal",
    "category": "Add Mesh",
    "description": "Add a fractal!",
    "author": "Wannes Malfait",
    "wiki": "https://en.wikipedia.org/wiki/L-system#Example_5:_Sierpinski_triangle",
    "version": (1, 0),
    "blender": (2, 80, 0),
}


class Params(bpy.types.PropertyGroup):
    """Parameters for the Fractal"""
    RemoveDoubles : bpy.props.BoolProperty(
        name = "Remove doubles",
        description = "Removes doubles after generating fractal (increases calculation time)")
    Variables : bpy.props.StringProperty(
        name = "Variables",
        description = "These will be updated according to the rules",
        default="AB",
        maxlen = 2)
    Constants : bpy.props.StringProperty(
        name = "Constants",
        description = "These remain unaffected by the rules",
        default="+-")
    Rule1 : bpy.props.StringProperty(
        name ="Rule 1",
        description="Rule for first variable",
        default="B-A-B")
    Rule2 : bpy.props.StringProperty(
        name = "Rule 2",
        description="Rule for second variable",
        default = "A+B+A")
    Axiom : bpy.props.StringProperty(
        name = "Axiom",
        description = "The initial state",
        default = "A")
    Iterations : bpy.props.IntProperty(
        name="Iterations",
        description="Amount of recursion depth in the fractal",
        default=4,
        min = 0,
        max = 15)
    Angle : bpy.props.FloatProperty(
        name="Angle",
        description="The angle each segment is turned by",
        default = 60,
        min = -360,
        max = 360)
    Length : bpy.props.FloatProperty(
        name="Length",
        description="The length of each segment",
        default = 0.1,
        min = 0.0001)


class MESH_OT_addFractal(bpy.types.Operator, bpy_extras.object_utils.AddObjectHelper):
    """Creates the Fractal"""
    bl_idname = "mesh.add_fractal"
    bl_label = "Add Fractal"
    bl_options = {'REGISTER', 'UNDO'}

    p : bpy.props.PointerProperty(type=Params)

    def execute(self, context):
        # mesh arrays
        verts = [(0,0,0)]
        faces = []
        edges = []

        
        #creating the lindemayer system:
        variables = self.p.Variables
        constants = self.p.Constants
        start = self.p.Axiom
        rules = [self.p.Rule1,self.p.Rule2]
        
        #Fractal parameters
        iterations = self.p.Iterations
        angle = math.radians(self.p.Angle)
        length = self.p.Length
        remove_doubles = self.p.RemoveDoubles
        
        code = start
        for i in range(iterations):
            codecopy = code[:] # make a copy of the code
            code = ""
            for symb in codecopy:
                index = variables.find(symb) # is it in the variables?
                if index != -1: # we need to update according to the rules
                    code += rules[index]
                else:
                    code+=symb # ok we keep the constant
        
        
        
        #fill verts array and connect with edges
        theta = 0 # we start by moving forward
        count = 0
        if  constants == "+-" or constants == "-+": #We use A and B as move forward
            for symb in code:
                if symb =="+":
                    theta += angle #turn right
                elif symb == "-":
                    theta -= angle #turn left
                else: # otherwise it means we have to add a new segment
                    x = verts[-1][0] + length*math.cos(theta) #we add a new segment
                    y = verts[-1][1] + length*math.sin(theta)
                    vert = (x, y, 0) 
                    verts.append(vert)
                    count+=1
                    edges.append((count-1,count)) #connect the new vertex
        elif ("<" in constants and ">" in constants) or ("[" in constants and "]" in constants):
            prev_locations = []
            prev_angles = []
            prevx = 0
            prevy = 0
            older_vert = False
            old_ind = 0
            for symb in code:
                if symb == "x" or symb == "X":
                    continue 
                elif symb =="+":
                    theta += angle #turn right
                elif symb == "-":
                    theta -= angle #turn left
                elif symb == "<" or symb == "[":
                    prev_locations.append((prevx,prevy,count))
                    prev_angles.append(theta)
                    theta-=angle
                elif symb == ">" or symb == "]":
                    vert = prev_locations.pop()
                    prevx = vert[0]
                    prevy = vert[1]
                    old_ind = vert[2]
                    older_vert = True
                    theta = prev_angles.pop()
                    theta+=angle  
                else: # otherwise we need to add a new segment
                    x = prevx + length*math.cos(theta) #we add a new segment
                    y = prevy + length*math.sin(theta)
                    vert = (x, y, 0) 
                    verts.append(vert)
                    count+=1
                    if(older_vert):
                        edges.append((old_ind,count)) #connect with an old point
                        older_vert = False
                    else:
                        edges.append((count-1,count)) #connect the new vertex
                    prevx = x
                    prevy = y
                    
        elif constants == "" and len(variables) >0:
            x_loc = 0
            y_loc = 0
            theta = angle
            if(len(variables)==1):
                variables+="B" #just so we don't get any errors
            for symb in code:
                if symb == variables[1]:
                    x_loc += length*math.cos(theta) #we move forward
                    y_loc += length*math.sin(theta)
                    vert = (x_loc, y_loc, 0) 
                    verts.append(vert)
                    count+=1
                else: # otherwise it means we have to add a new segment
                    x_loc += length*math.cos(theta) #we add a new segment
                    y_loc += length*math.sin(theta)
                    vert = (x_loc, y_loc, 0) 
                    verts.append(vert)
                    count+=1
                    edges.append((count-1,count)) #connect the new vertex
        else:
            for symb in code:
                if symb =="+":
                    theta += angle #turn right
                elif symb == "-":
                    theta -= angle #turn left
                elif symb in constants:
                    x = verts[-1][0] + length*math.cos(theta) #we add a new segment
                    y = verts[-1][1] + length*math.sin(theta)
                    vert = (x, y, 0) 
                    verts.append(vert)
                    count+=1
                    edges.append((count-1,count)) #connect the new vertex
        
                
        # create mesh and object
        mymesh = bpy.data.meshes.new("Fractal")
        
        
        # create mesh from python data
        mymesh.from_pydata(verts, edges, faces)
        
        
        #mymesh.update(calc_edges=True)
        bpy_extras.object_utils.object_data_add(context, mymesh, operator=self)
        
        #remove doubles?
        if remove_doubles:
            #go to editmode
            bpy.ops.object.editmode_toggle()
            
            # remove duplicate vertices
            bpy.ops.mesh.remove_doubles()
            #go back to objectmode
            bpy.ops.object.editmode_toggle()
            
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        col = layout.column(align=True)
        # Control over the parameters
        col.label(text="L-system:")
        col.prop(self.p, "Variables")
        col.prop(self.p, "Constants")
        col.prop(self.p, "Rule1")
        col.prop(self.p, "Rule2")
        col.prop(self.p, "Axiom")
        col.label(text="Parameters:")
        col.prop(self.p, "RemoveDoubles")
        col.prop(self.p, "Iterations")
        col.prop(self.p, "Angle")
        col.prop(self.p, "Length")


def addMenu(self, context):
    self.layout.operator(MESH_OT_addFractal.bl_idname,
                         text="Add Fractal")



#register, unregister = bpy.utils.register_classes_factory(classes)
def register():
    bpy.utils.register_class(Params)
    bpy.utils.register_class(MESH_OT_addFractal)
    bpy.types.VIEW3D_MT_mesh_add.append(addMenu)


def unregister():
    bpy.utils.unregister_class(Params)
    bpy.utils.unregister_class(MESH_OT_addFractal)
    bpy.types.VIEW3D_MT_mesh_add.remove(addMenu)


if __name__ == "__main__":
    register()
