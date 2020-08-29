

import bpy

bl_info = {
    "name": "Node Value Finder",
    "author": "Wannes Malfait",
    "version": (1, 0),
    "location": "Node Editor Toolbar",
    "description": "Find out what values of each node do",
    "warning": "",
    "category": "Node",
}

def find_inputs_cb(self,context):
    
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
        inputs.append((name,name,name))
    return inputs


class VF_Settings(bpy.types.PropertyGroup):
    steps : bpy.props.IntProperty(
        name = "Steps",
        description = "Number of inbetween frames",
        default = 40,
        min = 2,
        soft_max = 100,
        )
    input : bpy.props.EnumProperty(
        name = "Input",
        description = "Values are generated for this input",
        items=find_inputs_cb,
    )
    start_input : bpy.props.FloatProperty(
        name = 'Starting value',
        default = 0,
    )
    end_input : bpy.props.FloatProperty(
        name = 'End value',
        default = 1,
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
        space = context.space_data
        node_tree = space.node_tree
        node_active = context.active_node
        props = context.scene.value_finder 
        rd = context.scene.render
        
        input_socket = node_active.inputs[props.input] 
        original_path = rd.filepath
        for step in range(props.steps):
            # Steps is at least 2, so no zero division errors
            slope = (props.end_input-props.start_input)/(props.steps-1)
            value = props.start_input+slope*step
            input_socket.default_value = value
            rd.filepath = original_path + "-" + str(step+1) + "({} = {})".format(input_socket.name,value)
            bpy.ops.render.render(write_still = True)
        rd.filepath = original_path
        
        self.report({'INFO'}, "Finished Rendering")
        return {'FINISHED'}


class VF_PT_panel(bpy.types.Panel):
    bl_idname = "NODE_PT_vf_value_finder"
    bl_space_type = 'NODE_EDITOR'
    bl_label = "Value Finder"
    bl_region_type = "UI"
    bl_category = "Value Finder"
    
    def draw(self, context):
        
        #Helper variables
        layout = self.layout
        scene = context.scene
        props = scene.value_finder
        
        col = layout.column(align=True)
        
        
        if context.active_node is None:
            col.label(text='No active node')
            return
        node = context.active_node
        if len(node.inputs) ==  0:
            col.label(text='Active node has no input')
        elif len(node.outputs) ==  0:
            col.label(text='Active node has no output')
        else:
            
            col.prop(props, 'input')
            if props.input is None or props.input == '':
                col.label(text='Select an input')
                return
            input_socket = node.inputs[props.input]
            col.prop(props, 'steps')
            col.prop(props, 'start_input')
            col.prop(props, 'end_input')
            col.separator()
            #For Convenience
            layout.use_property_split = True
            col = layout.column(align=True)
            
            col.label(text="Output settings:")
            rd = scene.render
            col.prop(rd,'filepath')
            col.prop(rd, 'engine')
            col.prop(rd, "resolution_x", text="Resolution X")
            col.prop(rd, "resolution_y", text="Y")
            col.prop(rd, "resolution_percentage", text="%")
            col.separator()
        
            col.operator(VF_OT_value_finder.bl_idname, text="Generate Values", icon='OUTPUT')

classes = (
    VF_Settings,
    VF_OT_value_finder,
    VF_PT_panel,
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
