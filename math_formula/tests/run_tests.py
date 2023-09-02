import os

import bpy

import math_formula.backends.type_defs as td
from math_formula import ast_defs, file_loading
from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.shader_nodes import ShaderNodesBackEnd
from math_formula.compiler import Compiler
from math_formula.interpreter import Interpreter
from math_formula.mf_parser import Parser
from math_formula.type_checking import TypeChecker

if __name__ == "__main__":
    test_directory = os.path.dirname(os.path.realpath(__file__))
    filenames = os.listdir(test_directory)
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
        if filename.endswith("py"):
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

            # No errors
            num_succeeded += 1
    print(
        f"Tests finished {num_succeeded - num_skipped}/{tot_tests - num_skipped} succeeded"
    )
