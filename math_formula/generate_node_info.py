import itertools
import os
from typing import cast

import bpy


def generate_node_info() -> None:
    all_nodes: set[str] = set()

    for name in filter(lambda t: "Node" in t and "Group" not in t, dir(bpy.types)):
        if (
            name.startswith("Shader")
            or name.startswith("Geometry")
            or name.startswith("Function")
        ):
            all_nodes.add(name)

    # Add some "progress indicator" to the mouse cursor.
    # This gives feedback to the user that the script is
    # running and doing something.
    wm = bpy.context.window_manager
    wm.progress_begin(0, len(all_nodes))

    shader_tree = bpy.data.node_groups.new("TESTING_SHADERS", "ShaderNodeTree")
    geo_tree = bpy.data.node_groups.new("TESTING_GEOMETRY", "GeometryNodeTree")
    shader_tree.nodes.clear()
    geo_tree.nodes.clear()
    geo_nodes = geo_tree.nodes
    shader_nodes = shader_tree.nodes
    dtypes = {
        "VALUE": "FLOAT",
        "INT": "INT",
        "BOOLEAN": "BOOL",
        "VECTOR": "VEC3",
        "STRING": "STRING",
        "RGBA": "RGBA",
        "SHADER": "SHADER",
        "OBJECT": "OBJECT",
        "IMAGE": "IMAGE",
        "GEOMETRY": "GEOMETRY",
        "COLLECTION": "COLLECTION",
        "TEXTURE": "TEXTURE",
        "MATERIAL": "MATERIAL",
        "ROTATION": "ROTATION",
    }

    default_props = bpy.types.FunctionNode.bl_rna.properties  # type:ignore

    def snake(s: str) -> str:
        """Turn a string into snake case"""
        first_try = (
            "_".join(s.split())
            .replace("/", "_")
            .replace("-", "_")
            .replace("&", "_")
            .lower()
        )
        return "".join(
            [c for c in first_try if c.isalpha() or c.isnumeric() or c == "_"]
        )

    shader_geo_alias_strs = []
    geometry_alias_strs = []
    shader_alias_strs = []
    builtin_node_strs = []

    for i, node_name in enumerate(all_nodes):
        wm.progress_update(i)

        try:
            if node_name.startswith("Shader"):
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
            [
                f"('{snake(inp.name)}', DataType.{dtypes[cast(str,inp.type)]})"
                for inp in node.inputs
                if inp.type in dtypes
            ]
        )
        outputs = ", ".join(
            [
                f"('{snake(outp.name)}', DataType.{dtypes[cast(str, outp.type)]})"
                for outp in node.outputs
                if outp.type in dtypes
            ]
        )

        props = [
            cast(bpy.types.EnumProperty, prop)
            for prop in node.bl_rna.properties  # type: ignore
            if not prop.is_readonly
            and prop.type == "ENUM"
            and prop.identifier not in default_props
        ]

        # Generate aliases and validate properties
        default_name = snake(node.bl_label)
        if "legacy" in default_name:
            continue

        default_prop_values = [getattr(node, prop.identifier) for prop in props]

        def generate_alias(
            name: str, curr_state: list[str], permutation: tuple[int, ...]
        ) -> None:
            enabled_inputs = [i for i, input in enumerate(node.inputs) if input.enabled]
            enabled_outputs = [
                i for i, output in enumerate(node.outputs) if output.enabled
            ]
            property_values = [
                (prop.identifier, enum_val) for enum_val, prop in zip(curr_state, props)
            ]
            shuffled_property_values = [property_values[i] for i in permutation]
            alias_str = (
                f"'{name}' : "
                + f"NodeInstance('{node.bl_idname}', {enabled_inputs},"
                + f" {enabled_outputs}, {shuffled_property_values}), "
            )

            if node.bl_idname.startswith("Shader"):
                if supports_geometry_nodes:
                    shader_geo_alias_strs.append(alias_str)
                else:
                    shader_alias_strs.append(alias_str)
            elif node.bl_idname.startswith("Function") or node.bl_idname.startswith(
                "Geometry"
            ):
                geometry_alias_strs.append(alias_str)

        # Create an alias for every possible combination of enum values
        # that gives a valid result. Sadly, we can not determine which
        # variants actually "do something", just which ones don't crash.
        for combination in itertools.product(*[prop.enum_items for prop in props]):
            # Enum variants being set may disable other enum variants.
            # Because of this, we test every permutation of setting
            # the enums.
            # TODO: Is there some heuristic we can use to determine which
            # permutation is the "right one"? Now we just take the first
            # one that works.
            for permutation in itertools.permutations(range(len(combination))):
                name = default_name
                all_ok = True
                for i in permutation:
                    prop = props[i]
                    enum_value = combination[i]
                    try:
                        setattr(node, prop.identifier, enum_value.identifier)
                    except TypeError:
                        all_ok = False
                        break
                    # TODO: check that this is a valid name (no '&,' '/'...)
                    name += "_" + snake(enum_value.name)

                if all_ok:
                    generate_alias(
                        name,
                        [e.identifier for e in combination],
                        permutation,
                    )

                # Reset node to default state
                for prop, value in zip(reversed(props), reversed(default_prop_values)):
                    setattr(node, prop.identifier, value)

                if all_ok:
                    # No need to add other permutations
                    break

        # Generate the default case as well.
        if len(props) > 0:
            generate_alias(default_name, [], ())

        builtin_node_strs.append(
            f"'{node.bl_idname}' : BuiltinNode([{inputs}],\n\t\t[{outputs}]),"
        )
        if node_name.startswith("Shader"):
            shader_nodes.remove(node)
        else:
            geo_nodes.remove(node)

    bpy.data.node_groups.remove(shader_tree)
    bpy.data.node_groups.remove(geo_tree)

    add_on_dir = os.path.dirname(os.path.realpath(__file__))
    with open(
        os.path.join(os.path.join(add_on_dir, "backends"), "builtin_nodes.py"), "r+"
    ) as f:
        text = f.read()
        start_marker = "# Start auto generated\n"
        end_marker = "# End auto generated\n"
        start = text.find(start_marker) + len(start_marker)
        end = text.find(end_marker)
        builtin_node_strs.sort()
        shader_alias_strs.sort()
        shader_geo_alias_strs.sort()
        geometry_alias_strs.sort()
        builtin_node_str = "\n\t".join(builtin_node_strs)
        shader_geo_alias_str = "\n\t".join(shader_geo_alias_strs)
        geometry_alias_str = "\n\t".join(geometry_alias_strs)
        shader_alias_str = "\n\t".join(shader_alias_strs)
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

        wm.progress_end()

    import importlib

    from .backends import builtin_nodes, geometry_nodes
    from .backends import shader_nodes as shader_nodes_mod

    importlib.reload(builtin_nodes)
    importlib.reload(geometry_nodes)
    importlib.reload(shader_nodes_mod)
    for options_dict in [
        builtin_nodes.instances,
        geometry_nodes.geometry_nodes,
        shader_nodes_mod.shader_nodes,
    ]:
        for options in options_dict.values():
            for node_name_alias in options:
                if isinstance(node_name_alias, str):
                    if node_name_alias in builtin_nodes.geometry_node_aliases:
                        continue
                    if node_name_alias in builtin_nodes.shader_node_aliases:
                        continue
                    if node_name_alias in builtin_nodes.shader_geo_node_aliases:
                        continue
                    print("Invalid alias:", node_name_alias)


if __name__ == "__main__":
    generate_node_info()
