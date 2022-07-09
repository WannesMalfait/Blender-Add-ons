import copy
from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.shader_nodes import ShaderNodesBackEnd
from math_formula.backends.type_defs import *
from math_formula.backends.main import BackEnd
from math_formula.backends import builtin_nodes
from math_formula.parser import Error
from math_formula.type_checking import TypeChecker


class Compiler():

    @staticmethod
    def choose_backend(tree_type: str) -> BackEnd:
        if tree_type == 'GeometryNodeTree':
            return GeometryNodesBackEnd()
        elif tree_type == 'ShaderNodeTree':
            return ShaderNodesBackEnd()

    def __init__(self, tree_type: str) -> None:
        self.operations: list[Operation] = []
        self.errors: list[Error] = []
        self.back_end: BackEnd = self.choose_backend(tree_type)

    def compile(self, source: str) -> bool:
        type_checker = TypeChecker(self.back_end)
        succeeded = type_checker.type_check(source)
        typed_ast = type_checker.typed_repr
        self.errors = type_checker.errors
        if not succeeded:
            return False
        statements = typed_ast.body
        for statement in statements:
            if isinstance(statement, ty_expr):
                self.compile_expr(statement)
            elif isinstance(statement, TyAssign):
                self.compile_assign(statement)
            else:
                # These are the only possibilities for now
                assert False, "Unreachable code"
            self.operations.append(Operation(OpType.END_OF_STATEMENT, None))
        return True

    def compile_assign(self, assign: TyAssign):
        targets = assign.targets
        if isinstance(assign.value, Const):
            # Assignment to a value, so we need to create an input
            # node.
            assert len(targets) == 1, 'No structured assignment yet'
            if (target := targets[0]) is None:
                return
            value = assign.value.value
            dtype = assign.value.dtype[0]
            dtype = self.back_end.create_input(
                self.operations, target.id, value, dtype)
            return
        # Output will be some node socket, so just simple assignment
        self.compile_expr(assign.value)
        if len(targets) > 1:
            if assign.value.stype == StackType.STRUCT:
                self.operations.append(Operation(OpType.SPLIT_STRUCT, None))
            elif assign.value.dtype[0] == DataType.VEC3:
                self.operations.append(Operation(OpType.CALL_BUILTIN,
                                                 NodeInstance('ShaderNodeSeparateXYZ', [0], [0, 1, 2], [])))
                self.operations.append(Operation(OpType.SPLIT_STRUCT, None))
            elif assign.value.dtype[0] == DataType.RGBA:
                raise NotImplementedError
            else:
                assert False, 'Unreachable, bug in type checker'
        for target in targets:
            if target is None:
                continue
            self.operations.append(Operation(OpType.CREATE_VAR, target.id))

    def compile_expr(self, expr: ty_expr):
        if isinstance(expr, Const):
            self.const(expr)
        elif isinstance(expr, Var):
            self.var(expr)
        elif isinstance(expr, NodeCall):
            self.node_call(expr)
        elif isinstance(expr, GetOutput):
            self.get_output(expr)
        else:
            print(expr, type(expr))
            assert False, "Unreachable code"

    def node_call(self, expr: NodeCall):
        for arg in expr.args:
            self.compile_expr(arg)
            if arg.stype == StackType.STRUCT:
                # Get the output we need.
                self.operations.append(Operation(OpType.GET_OUTPUT, 0))
        # Add the implicit default arguments here
        for _ in range(len(expr.node.inputs) - len(expr.args)):
            self.operations.append(Operation(OpType.PUSH_VALUE, None))
        # To add the node we need the bl_name instead.
        expr.node = copy.copy(expr.node)
        expr.node.key = builtin_nodes.nodes[expr.node.key].bl_name
        self.operations.append(Operation(OpType.CALL_BUILTIN, expr.node))

    def const(self, const: Const):
        self.operations.append(Operation(OpType.PUSH_VALUE, const.value))

    def var(self, var: Var):
        # We should only end up here when we want to 'load' a variable.
        # If the name doesn't exist yet, create a default value
        if var.needs_instantion:
            self.back_end.create_input(
                self.operations, var.id, None, var.dtype[0])
        self.operations.append(Operation(OpType.GET_VAR, var.id))

    def get_output(self, get_output: GetOutput):
        self.compile_expr(get_output.value)
        self.operations.append(Operation(OpType.GET_OUTPUT, get_output.index))


if __name__ == '__main__':
    import os
    add_on_dir = os.path.dirname(
        os.path.realpath(__file__))
    test_directory = os.path.join(add_on_dir, 'tests')
    filenames = os.listdir(test_directory)
    verbose = 3
    num_passed = 0
    tot_tests = 0
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[96m'
    ENDC = '\033[0m'
    for filename in filenames:
        tot_tests += 1
        print(f'Testing: {BOLD}{filename}{ENDC}:  ', end='')
        with open(os.path.join(test_directory, filename), 'r') as f:
            compiler = Compiler('GeometryNodeTree')
            try:
                success = compiler.compile(f.read())
                print(GREEN + 'No internal errors' + ENDC)
                if verbose > 0:
                    print(
                        f'{YELLOW}Compiler errors{ENDC}' if not success else f'{BLUE}No compiler errors{ENDC}')
                if verbose > 1 and success:
                    print(compiler.errors)
                if verbose > 2:
                    print(*compiler.operations, sep='\n')
                num_passed += 1
            except NotImplementedError:
                print(RED + 'Internal errors' + ENDC)
    print(f'Tests done: Passed: ({num_passed}/{tot_tests})')
