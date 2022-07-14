from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.shader_nodes import ShaderNodesBackEnd
from math_formula.backends.type_defs import *
from math_formula.backends.main import BackEnd
from math_formula.parser import Error
from math_formula.type_checking import TypeChecker


class Compiler():

    @staticmethod
    def choose_backend(tree_type: str) -> BackEnd:
        if tree_type == 'GeometryNodeTree':
            return GeometryNodesBackEnd()
        elif tree_type == 'ShaderNodeTree':
            return ShaderNodesBackEnd()

    def __init__(self, tree_type: str, file_data: FileData = None) -> None:
        self.operations: list[Operation] = []
        self.errors: list[Error] = []
        self.back_end: BackEnd = self.choose_backend(tree_type)
        if file_data is not None:
            self.type_checker = TypeChecker(
                self.back_end, file_data.geometry_nodes if tree_type == 'GeometryNodeTree' else file_data.shader_nodes)
        else:
            self.type_checker = TypeChecker(self.back_end, {})
        self.curr_function: TyFunction = None

    def check_functions(self, source: str) -> bool:
        self.type_checker.type_check(source)
        self.errors = self.type_checker.errors
        return (self.errors == [])

    def compile(self, source: str) -> bool:
        succeeded = self.type_checker.type_check(source)
        typed_ast = self.type_checker.typed_repr
        self.errors = self.type_checker.errors
        if not succeeded:
            return False
        statements = typed_ast.body
        for statement in statements:
            self.compile_statement(statement)
        return True

    def compile_statement(self, stmt: ty_stmt):
        if isinstance(stmt, ty_expr):
            self.compile_expr(stmt)
        elif isinstance(stmt, TyAssign):
            self.compile_assign_like(stmt)
        elif isinstance(stmt, TyOut):
            self.compile_assign_like(stmt)
        else:
            # These are the only possibilities for now
            assert False, "Unreachable code"
        self.operations.append(Operation(OpType.END_OF_STATEMENT, None))

    def compile_assign_like(self, assign: Union[TyAssign, TyOut]):
        targets = assign.targets
        if isinstance(assign.value, Const):
            # Assignment to a value, so we need to create an input
            # node.
            if (target := targets[0]) is None:
                return
            value = assign.value.value
            dtype = assign.value.dtype[0]
            if isinstance(assign, TyAssign):
                self.back_end.create_input(
                    self.operations, target.id, value, dtype)
                self.operations.append(Operation(OpType.CREATE_VAR, target.id))
            else:
                self.back_end.create_input(
                    self.operations, self.curr_function.inputs[target].name, value, dtype)
                self.operations.append(
                    Operation(OpType.SET_FUNCTION_OUT, target))
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
        elif isinstance(assign, TyOut) and assign.value.stype == StackType.STRUCT:
            self.operations.append(Operation(OpType.GET_OUTPUT, 0))
        for target in targets:
            if target is None:
                continue
            if isinstance(assign, TyAssign):
                self.operations.append(Operation(OpType.CREATE_VAR, target.id))
            else:
                self.operations.append(
                    Operation(OpType.SET_FUNCTION_OUT, target))

    def compile_expr(self, expr: ty_expr):
        if isinstance(expr, Const):
            self.const(expr)
        elif isinstance(expr, Var):
            self.var(expr)
        elif isinstance(expr, NodeCall):
            self.node_call(expr)
        elif isinstance(expr, GetOutput):
            self.get_output(expr)
        elif isinstance(expr, FunctionCall):
            self.func_call(expr)
        else:
            print(expr, type(expr))
            assert False, "Unreachable code"

    def compile_function(self, func: TyFunction) -> CompiledFunction:
        outer_ops = self.operations
        self.operations = []
        for stmt in func.body:
            self.compile_statement(stmt)
        compiled_body = self.operations
        self.operations = outer_ops
        return CompiledFunction(
            [i.name for i in func.inputs], compiled_body, len(func.outputs))

    def compile_node_group(self, func: TyFunction) -> CompiledNodeGroup:
        outer_ops = self.operations
        self.operations = []
        for stmt in func.body:
            self.compile_statement(stmt)
        compiled_body = self.operations
        self.operations = outer_ops
        return CompiledNodeGroup(func.name, func.inputs, func.outputs, compiled_body)

    def func_call(self, expr: FunctionCall):
        for arg in expr.args:
            self.compile_expr(arg)
            if arg.stype == StackType.STRUCT:
                # Get the output we need.
                self.operations.append(Operation(OpType.GET_OUTPUT, 0))
        # Add the implicit default arguments here
        for i in range(len(expr.args), len(expr.function.inputs)):
            self.operations.append(
                Operation(OpType.PUSH_VALUE, expr.function.inputs[i].value))
        if expr.function.is_node_group:
            self.operations.append(
                Operation(OpType.CALL_NODEGROUP, self.compile_node_group(expr.function)))
            return
        self.operations.append(
            Operation(OpType.CALL_FUNCTION, self.compile_function(expr.function)))

    def node_call(self, expr: NodeCall):
        for arg in expr.args:
            self.compile_expr(arg)
            if arg.stype == StackType.STRUCT:
                # Get the output we need.
                self.operations.append(Operation(OpType.GET_OUTPUT, 0))
        # Add the implicit default arguments here
        for _ in range(len(expr.node.inputs) - len(expr.args)):
            self.operations.append(Operation(OpType.PUSH_VALUE, None))
        self.operations.append(Operation(OpType.CALL_BUILTIN, expr.node))

    def const(self, const: Const):
        self.operations.append(Operation(OpType.PUSH_VALUE, const.value))

    def var(self, var: Var):
        # We should only end up here when we want to 'load' a variable.
        # If the name doesn't exist yet, create a default value
        if var.needs_instantion:
            self.back_end.create_input(
                self.operations, var.id, None, var.dtype[0])
            self.operations.append(Operation(OpType.CREATE_VAR, var.id))
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
        # if filename != 'functions':
        #     continue
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
