import os

import bpy

import math_formula.backends.type_defs as td
from math_formula import ast_defs
from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.shader_nodes import ShaderNodesBackEnd
from math_formula.compiler import Compiler
from math_formula.interpreter import Interpreter
from math_formula.mf_parser import Parser
from math_formula.scanner import Scanner, TokenType
from math_formula.type_checking import TypeChecker

if __name__ == "__main__":
    test_directory = os.path.dirname(os.path.realpath(__file__))
    filenames = os.listdir(test_directory)
    tot_tests = 0
    num_succeeded = 0
    for filename in filenames:
        if filename.endswith("py"):
            continue
        print(f'\nTesting: "{filename}"')
        with open(os.path.join(test_directory, filename), "r") as f:
            source = f.read()

            scanner = Scanner(source)
            tokens = []
            while (token := scanner.scan_token()).token_type != TokenType.EOL:
                tokens.append(token)

            parser = Parser(source)
            mod = parser.parse()
            if parser.had_error:
                print("Parsing failed")
                print(ast_defs.dump(mod, indent="."))
                continue

            for tree_type in ("Shader", "Geometry"):
                if tree_type == "Shader":
                    type_checker = TypeChecker(ShaderNodesBackEnd())
                else:
                    type_checker = TypeChecker(GeometryNodesBackEnd())

                try:
                    success = type_checker.type_check(source)
                except NotImplementedError:
                    print("Implementation not yet finished, skipping test")
                    break

                if not success:
                    print(type_checker.errors)
                    print(ast_defs.dump(type_checker.typed_repr, td.ty_ast, indent="."))
                    continue

                node_tree = bpy.data.node_groups.new(
                    f"test_{filename}_{tree_type.lower()}", f"{tree_type}NodeTree"
                )

                compiler = Compiler(node_tree.bl_idname)
                compiler.compile(source)

                interpreter = Interpreter(node_tree)
                for operation in compiler.operations:
                    interpreter.operation(operation)
