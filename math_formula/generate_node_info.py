import bpy
from typing import cast

print('\n\nSTARTING\n\n')
all_nodes: set[str] = set()


for name in filter(lambda t: 'Node' in t and not 'Group' in t, dir(bpy.types)):
    if name.startswith('Shader') or name.startswith('Geometry') or name.startswith('Function'):
        all_nodes.add(name)

shader_tree = bpy.data.materials['Material'].node_tree
geo_tree = bpy.data.node_groups['Geometry Nodes']
geo_nodes = geo_tree.nodes
shader_nodes = shader_tree.nodes
dtypes = {
    'VALUE': 'FLOAT',
    'INT': 'INT',
    'BOOLEAN': 'BOOL',
    'VECTOR': 'VEC3',
    'STRING': 'STRING',
    'RGBA': 'RGBA',
    'SHADER': 'SHADER',
    'OBJECT': 'OBJECT',
    'IMAGE': 'IMAGE',
    'GEOMETRY': 'GEOMETRY',
    'COLLECTION': 'COLLECTION',
    'TEXTURE': 'TEXTURE',
    'MATERIAL': 'MATERIAL',
}

default_props = bpy.types.FunctionNode.bl_rna.properties


def snake(s: str) -> str:
    """Turn a string into snake case"""
    return '_'.join(s.split()).lower()


aliases = {}

for node_name in all_nodes:
    try:
        if node_name.startswith('Shader'):
            node = shader_nodes.new(node_name)
        else:
            node = geo_nodes.new(node_name)
    except:
        continue
    inputs = ", ".join(
        [f"('{snake(inp.name)}', DataType.{dtypes[cast(str,inp.type)]})" for inp in node.inputs])
    outputs = ", ".join(
        [f"('{snake(outp.name)}', DataType.{dtypes[cast(str, outp.type)]})" for outp in node.outputs])

    props = [cast(bpy.types.EnumProperty, prop) for prop in node.bl_rna.properties if not prop.is_readonly and
             prop.type == 'ENUM' and not prop.identifier in default_props]

    # Generate aliases and validate properties
    default_name = snake(node.name)
    curr_state = [0 for _ in props]

    # Monstrosity needed because recursion is needed to go over all combinations.
    def rec(prop_i: int, name: str):
        if prop_i == len(props):
            # Final property was chosen. Store this combination.
            enabled_inputs = [i for i, input in enumerate(
                node.inputs) if input.enabled]
            enabled_outputs = [i for i, output in enumerate(
                node.outputs) if output.enabled]
            property_values = [(prop.identifier, prop.enum_items[enum_j].identifier)
                               for enum_j, prop in zip(curr_state, props)]
            aliases[name] = [enabled_inputs, enabled_outputs, property_values]
            return

        # We still have a choice for the value of the next property.
        for enum_j in range(len(props[prop_i].enum_items)):
            curr_state[prop_i] = enum_j
            prop = props[prop_i]
            enum_value = prop.enum_items[enum_j]
            try:
                setattr(node, prop.identifier,
                        enum_value.identifier)
                rec(prop_i+1, name + '_' + snake(enum_value.name))
            except:
                # This property can't actually be set.
                continue

    # Recursively generate all posibilities.
    rec(0, default_name)

    # TODO: actually store this in a file.
    print(
        f"'{node.bl_idname}' : BuiltinNode([{inputs}],\n\t[{outputs}]),")
    if node_name.startswith('Shader'):
        shader_nodes.remove(node)
    else:
        geo_nodes.remove(node)

# TODO: Store aliases.
