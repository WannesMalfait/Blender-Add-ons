import os
import pickle
import typing

import bpy

import math_formula.backends.type_defs as td
from math_formula import ast_defs, file_loading
from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.shader_nodes import ShaderNodesBackEnd
from math_formula.compiler import Compiler
from math_formula.interpreter import Interpreter
from math_formula.mf_parser import Parser
from math_formula.type_checking import TypeChecker


class SimplifiedSocket:
    def __init__(self, value, link: typing.Optional["SimplifiedLink"] = None):
        self.value = value
        try:
            # Try to unpack vec3/vec4
            self.value = [v for v in self.value]
        except TypeError:
            pass
        self.link = link

    def __str__(self) -> str:
        return str(self.value) if self.link is None else str(self.link)

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False
        if self.link is not None:
            return self.link == other.link
        return self.value == other.value


class SimplifiedNode:
    def __init__(
        self,
        identifier: str,
        type: str,
        props: dict[str, typing.Any],
        in_sockets: list[SimplifiedSocket],
        out_sockets: list[SimplifiedSocket],
        node_tree: typing.Optional["SimplifiedNodeTree"] = None,
    ):
        self.identifier = identifier
        self.type = type
        self.props = props
        self.in_sockets = in_sockets
        self.out_sockets = out_sockets
        self.node_tree = node_tree

    def __str__(self) -> str:
        return (
            f"{self.identifier} ({self.type})\n"
            + f"{self.props}\n"
            + f"inputs: {self.in_sockets}\n"
            + f"outputs: {self.out_sockets}\n"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False
        return (
            self.identifier == other.identifier
            and self.type == other.type
            and self.props == other.props
            and self.in_sockets == other.in_sockets
            and self.out_sockets == other.out_sockets
            and self.node_tree == other.node_tree
        )


class SimplifiedLink:
    def __init__(
        self,
        from_node: SimplifiedNode,
        to_node: SimplifiedNode,
        from_socket: int,
        to_socket: int,
    ):
        self.from_node = from_node
        self.to_node = to_node
        self.from_socket = from_socket
        self.to_socket = to_socket

    def __str__(self) -> str:
        return (
            f"{self.from_node.identifier}[{self.from_socket}]"
            + f" -> {self.to_node.identifier}[{self.to_socket}]"
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False
        return (
            other.from_node.identifier == self.from_node.identifier
            and other.to_node.identifier == self.to_node.identifier
            and other.from_socket == self.from_socket
            and other.to_socket == self.to_socket
        )


class SimplifiedNodeTree:
    def __init__(self, node_tree: bpy.types.NodeTree):
        # We don't need to store the links explicitly,
        # as these can be found in the relevant sockets
        self.nodes: dict[str, SimplifiedNode] = {}
        default_props = bpy.types.FunctionNode.bl_rna.properties  # type:ignore
        for node in node_tree.nodes:
            props = {}
            for prop in node.bl_rna.properties:  # type:ignore
                if (
                    not prop.is_readonly
                    and prop.type == "ENUM"
                    and prop.identifier not in default_props
                ):
                    props[prop.identifier] = node.__getattribute__(prop.identifier)

            in_sockets = [
                SimplifiedSocket(
                    i.default_value  # type:ignore
                )
                for i in node.inputs
            ]
            out_sockets = [
                SimplifiedSocket(
                    o.default_value  # type:ignore
                )
                for o in node.outputs
            ]
            node_group_tree: typing.Optional["SimplifiedNodeTree"] = None
            if isinstance(node, bpy.types.NodeGroup):
                node_group_tree = SimplifiedNodeTree(node.node_tree)

            self.nodes[node.name] = SimplifiedNode(
                node.name,
                node.bl_idname,
                props,
                in_sockets,
                out_sockets,
                node_group_tree,
            )

        for link in node_tree.links:
            from_node = self.nodes[link.from_node.name]
            to_node = self.nodes[link.to_node.name]
            from_socket = to_socket = -1
            for i, s in enumerate(link.from_node.outputs):
                if s == link.from_socket:
                    from_socket = i
                    break
            for i, s in enumerate(link.to_node.inputs):
                if s == link.to_socket:
                    to_socket = i
                    break
            slink = SimplifiedLink(from_node, to_node, from_socket, to_socket)
            from_node.out_sockets[from_socket].link = slink
            to_node.in_sockets[to_socket].link = slink

    def __repr__(self) -> str:
        return str(self.nodes)

    def __eq__(self, other) -> bool:
        if type(other) != type(self):
            return False

        if len(self.nodes) != len(other.nodes):
            return False
        for node in self.nodes.values():
            other_node = other.nodes[node.identifier]
            if node != other_node:
                print("Resulting tree changed!".upper())
                print("Mismatch between nodes!!")
                print("Old node:", other_node)
                print("New node:", node)
                return False
        return True


if __name__ == "__main__":
    test_directory = os.path.dirname(os.path.realpath(__file__))
    filenames = os.listdir(test_directory)

    filenames.sort()

    tot_tests = 0
    num_succeeded = 0
    num_skipped = 0

    print("\nStarting tests:\n")
    errors = file_loading.load_custom_implementations(
        None, dir=file_loading.custom_implementations_dir, force_update=True
    )
    for filename, file_errors in errors:
        print(f"Errors in {filename}")
        for file_error in file_errors:
            print(file_error)
    if errors != []:
        print("Aborting due to errors in custom implementations")
        exit()

    for filename in filenames:
        if filename.endswith("py") or filename.endswith("out"):
            continue
        print(f'Testing: "{filename}"')
        with open(os.path.join(test_directory, filename), "r") as f:
            tot_tests += 1
            source = f.read()

            # Test the parser
            parser = Parser(source)
            mod = parser.parse()
            if parser.had_error:
                print("Parsing failed")
                print(ast_defs.dump(mod, indent="."))
                continue

            # Test the backend-specific code
            success = True
            for tree_type in ("Shader", "Geometry"):
                if tree_type == "Shader":
                    type_checker = TypeChecker(
                        ShaderNodesBackEnd(), file_loading.file_data.shader_nodes
                    )
                else:
                    type_checker = TypeChecker(
                        GeometryNodesBackEnd(), file_loading.file_data.geometry_nodes
                    )

                try:
                    success = type_checker.type_check(source)
                except NotImplementedError:
                    print("Implementation not yet finished, skipping test")
                    num_skipped += 1
                    success = False
                    break

                if not success:
                    print(type_checker.errors)
                    print(ast_defs.dump(type_checker.typed_repr, td.ty_ast, indent="."))
                    continue

                node_tree = bpy.data.node_groups.new(
                    f"test_{filename}_{tree_type.lower()}", f"{tree_type}NodeTree"
                )

                compiler = Compiler(node_tree.bl_idname, file_loading.file_data)
                compiler.compile(source)

                interpreter = Interpreter(node_tree)
                for operation in compiler.operations:
                    interpreter.operation(operation)

                # Do we have a previous output already?
                output_path = os.path.join(
                    test_directory, filename + "." + tree_type.lower() + "_out"
                )
                tree = SimplifiedNodeTree(node_tree)
                if os.path.exists(output_path):
                    # Compare the two node trees
                    with open(output_path, "rb") as f:  # type: ignore
                        other_tree = pickle.load(f)  # type: ignore
                        if tree != other_tree:
                            success = False
                            break
                else:
                    with open(output_path, "wb") as f:  # type: ignore
                        pickle.dump(tree, f)  # type: ignore

            # No errors
            if success:
                num_succeeded += 1
    print(f"Tests finished {num_succeeded}/{tot_tests - num_skipped} succeeded")
