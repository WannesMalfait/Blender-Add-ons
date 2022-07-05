from math_formula.nodes.base import ValueType, NodeFunction
from math_formula.nodes import shading as shader_nodes
from math_formula.nodes import geometry as geometry_nodes
from math_formula.nodes import functions as function_nodes
from math_formula.parser import Parser, Error
from math_formula import ast_defs
from bpy.types import NodeSocket
from typing import Union
from enum import IntEnum, auto


class OpType(IntEnum):
    # Push the given value on the stack. None represents a default value.
    PUSH_VALUE = 0
    # Create a variable with the given name, and assign it to stack.pop().
    CREATE_VAR = auto()
    # Get the variable with the given name, and push it onto the stack.
    GET_VAR = auto()
    # Get the output with the given index from the last value on the stack.
    # Put this value on top of the stack.
    GET_OUTPUT = auto()
    # Call the given function, all the arguments are on the stack. The value
    # on top of the stack is a list of the inputs for which arguments are
    # provided. Push the output onto the stack.
    CALL_FUNCTION = auto()
    # Same as CALL_FUNCTION except the function is just a built-in node.
    CALL_BUILTIN = auto()
    # Same as CALL_FUNCTION but a node group is created.
    CALL_NODEGROUP = auto()
    # Create an input node for the given type. The value is stack.pop().
    CREATE_INPUT = auto()
    # Clear the stack.
    END_OF_STATEMENT = auto()


class Operation():
    def __init__(self, op_type: OpType,
                 data: Union[ValueType, NodeFunction]) -> None:
        self.op_type = op_type
        self.data = data

    def __str__(self) -> str:
        return f"({self.op_type.name}, {self.data})"

    def __repr__(self) -> str:
        return self.__str__()


class Compiler():
    def __init__(self) -> None:
        self.operations: list[Operation] = []
        self.errors: list[Error] = []

    def compile(self, source: str, tree_type: str) -> bool:
        self.instructions = []
        parser = Parser(source)
        ast = parser.parse()
        self.errors = parser.errors
        if parser.had_error:
            return False
        statements = ast.body
        for statement in statements:
            if isinstance(statement, ast_defs.expr):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.FunctionDef):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.NodegroupDef):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.Out):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.Assign):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.Loop):
                raise NotImplementedError
            else:
                assert False, "Unreachable code"


# if __name__ == '__main__':
#     import os
#     add_on_dir = os.path.dirname(
#         os.path.realpath(__file__))
#     test_directory = os.path.join(add_on_dir, 'tests')
#     filenames = os.listdir(test_directory)
#     verbose = 1
#     num_passed = 0
#     tot_tests = 0
#     BOLD = '\033[1m'
#     GREEN = '\033[92m'
#     RED = '\033[91m'
#     YELLOW = '\033[93m'
#     BLUE = '\033[96m'
#     ENDC = '\033[0m'
#     for filename in filenames:
#         tot_tests += 1
#         print(f'Testing: {BOLD}{filename}{ENDC}:  ', end='')
#         with open(os.path.join(test_directory, filename), 'r') as f:
#             compiler = Compiler()
#             try:
#                 success = compiler.compile(f.read(), 'GeometryNodeTree')
#                 print(GREEN + 'No internal errors' + ENDC)
#                 if verbose > 0:
#                     print(
#                         f'{YELLOW}Syntax errors{ENDC}' if success else f'{BLUE}No syntax errors{ENDC}')
#                 if verbose > 1 and success:
#                     print(compiler.errors)
#                 if verbose > 2:
#                     print(compiler.operations)
#                 num_passed += 1
#             except NotImplementedError:
#                 print(RED + 'Internal errors' + ENDC)
#                 # if verbose > 0:
#                 #     print(
#                 #         f'{YELLOW}Syntax errors{ENDC}:' if compiler.parser.had_error else f'{BLUE}No syntax errors{ENDC}')
#                 # if verbose > 1 and compiler.parser.had_error:
#                 #     print(compiler.operations)
#     print(f'Tests done: Passed: ({num_passed}/{tot_tests})')
