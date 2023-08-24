from .backends import type_defs as td
from .backends.geometry_nodes import GeometryNodesBackEnd
from .backends.main import BackEnd
from .backends.shader_nodes import ShaderNodesBackEnd
from .mf_parser import Error
from .type_checking import TypeChecker


class Compiler():

    @staticmethod
    def choose_backend(tree_type: str) -> BackEnd:
        if tree_type == 'GeometryNodeTree':
            return GeometryNodesBackEnd()
        elif tree_type == 'ShaderNodeTree':
            return ShaderNodesBackEnd()
        else:
            assert False, 'Unreachable, compiler should not be used for other trees.'

    def __init__(self, tree_type: str, file_data: td.FileData | None = None) -> None:
        self.operations: list[td.Operation] = []
        self.errors: list[Error] = []
        self.back_end: BackEnd = self.choose_backend(tree_type)
        if file_data is not None:
            self.type_checker = TypeChecker(
                self.back_end, file_data.geometry_nodes if tree_type == 'GeometryNodeTree' else file_data.shader_nodes)
        else:
            self.type_checker = TypeChecker(self.back_end, {})
        self.curr_function: td.TyFunction | None = None

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

    def compile_statement(self, stmt: td.ty_stmt):
        if isinstance(stmt, td.ty_expr):
            self.compile_expr(stmt)
        elif isinstance(stmt, td.TyAssign):
            self.compile_assign_like(stmt)
        elif isinstance(stmt, td.TyOut):
            self.compile_assign_like(stmt)
        else:
            # These are the only possibilities for now
            assert False, "Unreachable code"
        self.operations.append(td.Operation(td.OpType.END_OF_STATEMENT, None))

    def compile_assign_like(self, assign: td.TyAssign | td.TyOut):
        targets = assign.targets
        if isinstance(assign.value, td.Const):
            # Assignment to a value, so we need to create an input
            # node.
            if (target := targets[0]) is None:
                return
            value = assign.value.value
            dtype = assign.value.dtype[0]
            if isinstance(assign, td.TyAssign):
                assert isinstance(
                    target, td.Var), 'Assignment should be to a Var'
                self.back_end.create_input(
                    self.operations, target.id, value, dtype)
                self.operations.append(td.Operation(
                    td.OpType.CREATE_VAR, target.id))
            else:
                assert isinstance(target, int) and isinstance(
                    assign, td.TyOut) and self.curr_function is not None
                self.back_end.create_input(
                    self.operations, self.curr_function.outputs[target].name, value, dtype)
                self.operations.append(
                    td.Operation(td.OpType.SET_FUNCTION_OUT, target))
            return
        # Output will be some node socket, so just simple assignment
        self.compile_expr(assign.value)
        if len(targets) > 1:
            if assign.value.stype == td.StackType.STRUCT:
                self.operations.append(td.Operation(
                    td.OpType.SPLIT_STRUCT, None))
            elif assign.value.dtype[0] == td.DataType.VEC3:
                self.operations.append(td.Operation(td.OpType.CALL_BUILTIN,
                                                    td.NodeInstance('ShaderNodeSeparateXYZ', [0], [0, 1, 2], [])))
                self.operations.append(td.Operation(
                    td.OpType.SPLIT_STRUCT, None))
            elif assign.value.dtype[0] == td.DataType.RGBA:
                raise NotImplementedError
            else:
                assert False, 'Unreachable, bug in type checker'
        elif isinstance(assign, td.TyOut) and assign.value.stype == td.StackType.STRUCT:
            self.operations.append(td.Operation(td.OpType.GET_OUTPUT, 0))
        for target in targets:
            if target is None:
                continue
            if isinstance(assign, td.TyAssign):
                assert isinstance(
                    target, td.Var), 'Assignment should be to a Var'
                self.operations.append(td.Operation(
                    td.OpType.CREATE_VAR, target.id))
            else:
                self.operations.append(
                    td.Operation(td.OpType.SET_FUNCTION_OUT, target))

    def compile_expr(self, expr: td.ty_expr):
        if isinstance(expr, td.Const):
            self.const(expr)
        elif isinstance(expr, td.Var):
            self.var(expr)
        elif isinstance(expr, td.NodeCall):
            self.node_call(expr)
        elif isinstance(expr, td.GetOutput):
            self.get_output(expr)
        elif isinstance(expr, td.FunctionCall):
            self.func_call(expr)
        else:
            print(expr, type(expr))
            assert False, "Unreachable code"

    def compile_function(self, func: td.TyFunction) -> td.CompiledFunction:
        outer_ops = self.operations
        self.operations = []
        self.curr_function = func
        for stmt in func.body:
            self.compile_statement(stmt)
        compiled_body = self.operations
        self.operations = outer_ops
        return td.CompiledFunction(
            [i.name for i in func.inputs], compiled_body, len(func.outputs))

    def compile_node_group(self, func: td.TyFunction) -> td.CompiledNodeGroup:
        outer_ops = self.operations
        self.operations = []
        self.curr_function = func
        for stmt in func.body:
            self.compile_statement(stmt)
        compiled_body = self.operations
        self.operations = outer_ops
        return td.CompiledNodeGroup(func.name, func.inputs, func.outputs, compiled_body)

    def func_call(self, expr: td.FunctionCall):
        for arg in expr.args:
            self.compile_expr(arg)
            if arg.stype == td.StackType.STRUCT:
                # Get the output we need.
                self.operations.append(td.Operation(td.OpType.GET_OUTPUT, 0))
        # Add the implicit default arguments here
        for i in range(len(expr.args), len(expr.function.inputs)):
            self.operations.append(
                td.Operation(td.OpType.PUSH_VALUE, expr.function.inputs[i].value))
        if expr.function.is_node_group:
            self.operations.append(
                td.Operation(td.OpType.CALL_NODEGROUP, self.compile_node_group(expr.function)))
            return
        self.operations.append(
            td.Operation(td.OpType.CALL_FUNCTION, self.compile_function(expr.function)))

    def node_call(self, expr: td.NodeCall):
        for arg in expr.args:
            self.compile_expr(arg)
            if arg.stype == td.StackType.STRUCT:
                # Get the output we need.
                self.operations.append(td.Operation(td.OpType.GET_OUTPUT, 0))
        # Add the implicit default arguments here
        for _ in range(len(expr.node.inputs) - len(expr.args)):
            self.operations.append(td.Operation(td.OpType.PUSH_VALUE, None))
        self.operations.append(td.Operation(td.OpType.CALL_BUILTIN, expr.node))

    def const(self, const: td.Const):
        self.operations.append(td.Operation(td.OpType.PUSH_VALUE, const.value))

    def var(self, var: td.Var):
        # We should only end up here when we want to 'load' a variable.
        # If the name doesn't exist yet, create a default value
        if var.needs_instantion:
            self.back_end.create_input(
                self.operations, var.id, None, var.dtype[0])
            self.operations.append(td.Operation(td.OpType.CREATE_VAR, var.id))
        self.operations.append(td.Operation(td.OpType.GET_VAR, var.id))

    def get_output(self, get_output: td.GetOutput):
        self.compile_expr(get_output.value)
        self.operations.append(td.Operation(
            td.OpType.GET_OUTPUT, get_output.index))


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
