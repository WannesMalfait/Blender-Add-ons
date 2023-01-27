import os
import bpy
from typing import cast


def generate_node_info():
    all_nodes: set[str] = set()

    for name in filter(lambda t: 'Node' in t and not 'Group' in t, dir(bpy.types)):
        if name.startswith('Shader') or name.startswith('Geometry') or name.startswith('Function'):
            all_nodes.add(name)

    shader_tree = bpy.data.node_groups.new('TESTING_SHADERS', 'ShaderNodeTree')
    geo_tree = bpy.data.node_groups.new('TESTING_GEOMETRY', 'GeometryNodeTree')
    shader_tree.nodes.clear()
    geo_tree.nodes.clear()
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
        first_try = '_'.join(s.split()).replace(
            '/', '_').replace('-', '_').replace('&', '_').lower()
        return ''.join([c for c in first_try if c.isalpha() or c.isnumeric() or c == '_'])

    shader_geo_alias_strs = []
    geometry_alias_strs = []
    shader_alias_strs = []
    builtin_node_strs = []

    for node_name in all_nodes:
        try:
            if node_name.startswith('Shader'):
                node = shader_nodes.new(node_name)
            else:
                node = geo_nodes.new(node_name)
        except:
            continue

        supports_geometry_nodes = False
        try:
            tmp = geo_nodes.new(node_name)
            geo_nodes.remove(tmp)
            supports_geometry_nodes = True
        except:
            pass
        inputs = ", ".join(
            [f"('{snake(inp.name)}', DataType.{dtypes[cast(str,inp.type)]})" for inp in node.inputs])
        outputs = ", ".join(
            [f"('{snake(outp.name)}', DataType.{dtypes[cast(str, outp.type)]})" for outp in node.outputs])

        props = [cast(bpy.types.EnumProperty, prop) for prop in node.bl_rna.properties if not prop.is_readonly and
                 prop.type == 'ENUM' and not prop.identifier in default_props]

        # Generate aliases and validate properties
        default_name = snake(node.bl_label)
        if 'legacy' in default_name:
            continue
        curr_state = [0 for _ in props]

        def generate_alias(name: str, curr_state: list[int]) -> None:
            enabled_inputs = [i for i, input in enumerate(
                node.inputs) if input.enabled]
            enabled_outputs = [i for i, output in enumerate(
                node.outputs) if output.enabled]
            property_values = [(prop.identifier, prop.enum_items[enum_j].identifier)
                               for enum_j, prop in zip(curr_state, props)]
            alias_str = f"'{name}' : NodeInstance('{node.bl_idname}', {enabled_inputs}, {enabled_outputs}, {property_values}),"

            if node.bl_idname.startswith('Shader'):
                if supports_geometry_nodes:
                    shader_geo_alias_strs.append(alias_str)
                else:
                    shader_alias_strs.append(alias_str)
            elif node.bl_idname.startswith('Function') or node.bl_idname.startswith('Geometry'):
                geometry_alias_strs.append(alias_str)

        # Monstrosity needed because recursion is needed to go over all combinations.
        def rec(prop_i: int, name: str):
            if prop_i == len(props):
                # Final property was chosen. Store this combination.
                generate_alias(name, curr_state)
                return

            # We still have a choice for the value of the next property.
            for enum_j in range(len(props[prop_i].enum_items)):
                curr_state[prop_i] = enum_j
                prop = props[prop_i]
                enum_value = prop.enum_items[enum_j]
                try:
                    prev_value = getattr(node, prop.identifier)
                    setattr(node, prop.identifier,
                            enum_value.identifier)
                except:
                    # This property can't actually be set.
                    # TODO: Check other permutations of setting this property.
                    continue
                # TODO: check that this is a valid name (no '&,' '/'...)
                rec(prop_i+1, name + '_' + snake(enum_value.name))
                setattr(node, prop.identifier, prev_value)

        # Generate the default case as well.
        if len(props) > 0:
            generate_alias(default_name, curr_state=[])
        # Recursively generate all posibilities.
        rec(0, default_name)

        builtin_node_strs.append(
            f"'{node.bl_idname}' : BuiltinNode([{inputs}],\n\t\t[{outputs}]),")
        if node_name.startswith('Shader'):
            shader_nodes.remove(node)
        else:
            geo_nodes.remove(node)

    bpy.data.node_groups.remove(shader_tree)
    bpy.data.node_groups.remove(geo_tree)

    add_on_dir = os.path.dirname(
        os.path.realpath(__file__))
    with open(os.path.join(os.path.join(add_on_dir, 'backends'), 'builtin_nodes.py'), 'r+') as f:
        text = f.read()
        start_marker = '# Start auto generated\n'
        end_marker = '# End auto generated\n'
        start = text.find(start_marker) + len(start_marker)
        end = text.find(end_marker)
        builtin_node_strs.sort()
        shader_alias_strs.sort()
        shader_geo_alias_strs.sort()
        geometry_alias_strs.sort()
        builtin_node_str = '\n\t'.join(builtin_node_strs)
        shader_geo_alias_str = '\n\t'.join(shader_geo_alias_strs)
        geometry_alias_str = '\n\t'.join(geometry_alias_strs)
        shader_alias_str = '\n\t'.join(shader_alias_strs)
        generated = f"""
# fmt: off
nodes = {{
\t{builtin_node_str}
}}

shader_geo_node_aliases = {{
\t{shader_geo_alias_str}
}}

geometry_node_aliases = {{
\t{geometry_alias_str}
}}

shader_node_aliases = {{
\t{shader_alias_str}
}}

# fmt: on

"""
        new_text = text[:start] + generated + text[end:]
        f.seek(0)
        f.write(new_text)
        f.truncate(len(new_text))

    import importlib
    from .backends import builtin_nodes, geometry_nodes, shader_nodes
    importlib.reload(builtin_nodes)
    importlib.reload(geometry_nodes)
    importlib.reload(shader_nodes)
    for options_dict in [builtin_nodes.instances, geometry_nodes.geometry_nodes, shader_nodes.shader_nodes]:
        for options in options_dict.values():
            for node_name in options:
                if isinstance(node_name, str):
                    if node_name in builtin_nodes.geometry_node_aliases:
                        continue
                    if node_name in builtin_nodes.shader_node_aliases:
                        continue
                    if node_name in builtin_nodes.shader_geo_node_aliases:
                        continue
                    print('Invalid alias:', node_name)


bpy.app.timers.register(generate_node_info, first_interval=0)

if __name__ == '__main__':
    generate_node_info()
