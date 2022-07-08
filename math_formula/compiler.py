from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
from math_formula.backends.main import BackEnd, DataType, Operation, OpType
from math_formula.parser import Parser, Error
from math_formula import ast_defs


class Compiler():
    def __init__(self, back_end: BackEnd) -> None:
        self.operations: list[Operation] = []
        self.errors: list[Error] = []
        self.curr_type: DataType = None
        self.back_end: BackEnd = back_end
        self.vars: dict[str, DataType] = {}

    def compile(self, source: str) -> bool:
        parser = Parser(source)
        ast = parser.parse()
        self.errors = parser.errors
        if parser.had_error:
            return False
        statements = ast.body
        for statement in statements:
            if isinstance(statement, ast_defs.expr):
                self.compile_expr(statement)
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
            self.operations.append(Operation(OpType.END_OF_STATEMENT, None))
        return True

    def compile_expr(self, expr: ast_defs.expr):
        if isinstance(expr, ast_defs.UnaryOp):
            self.unary_op(expr)
        elif isinstance(expr, ast_defs.BinOp):
            self.bin_op(expr)
        elif isinstance(expr, ast_defs.Constant):
            self.constant(expr)
        elif isinstance(expr, ast_defs.Vec3):
            self.vec3(expr)
        elif isinstance(expr, ast_defs.Rgba):
            raise NotImplementedError
        elif isinstance(expr, ast_defs.Name):
            self.name(expr)
        elif isinstance(expr, ast_defs.Attribute):
            raise NotImplementedError
        elif isinstance(expr, ast_defs.Keyword):
            raise NotImplementedError
        elif isinstance(expr, ast_defs.Call):
            raise NotImplementedError
        else:
            print(expr, type(expr))
            assert False, "Unreachable code"

    def compile_function(self, name: str, args: list[DataType], stack_locs: list[int]):
        dtype = self.back_end.compile_function(
            self.operations, name, args, stack_locs)
        self.curr_type = dtype

    def unary_op(self, un_op: ast_defs.UnaryOp):
        op = un_op.op
        self.compile_expr(un_op.operand)
        dtype = self.curr_type
        loc = len(self.operations) - 1
        if isinstance(op, ast_defs.Not):
            self.compile_function('not', [dtype], [loc])
        elif isinstance(op, ast_defs.USub):
            self.operations.append(Operation(OpType.PUSH_VALUE, -1))
            self.compile_function('mul', [DataType.INT, dtype], [loc])
        else:
            assert False, "Unreachable code"

    def bin_op(self, bin_op: ast_defs.BinOp):
        op = bin_op.op
        self.compile_expr(bin_op.left)
        l_dtype = self.curr_type
        l_loc = len(self.operations)-1
        self.compile_expr(bin_op.right)
        r_dtype = self.curr_type
        r_loc = len(self.operations) - 1
        if isinstance(op, ast_defs.And):
            self.compile_function('and', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Or):
            self.compile_function('or', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Add):
            self.compile_function('add', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Div):
            self.compile_function('div', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Mod):
            self.compile_function('mod', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Mult):
            self.compile_function('mul', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Pow):
            self.compile_function('pow', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Sub):
            self.compile_function('sub', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Eq):
            self.compile_function('equal', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Gt):
            self.compile_function(
                'greater_than', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.GtE):
            self.compile_function(
                'greater_equal', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.Lt):
            self.compile_function(
                'less_than', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.LtE):
            self.compile_function(
                'less_equal', [l_dtype, r_dtype], [l_loc, r_loc])
        elif isinstance(op, ast_defs.NotEq):
            self.compile_function(
                'not_equal', [l_dtype, r_dtype], [l_loc, r_loc])
        else:
            assert False, "Unreachable code"

    def constant(self, const: ast_defs.Constant):
        value, dtype = self.back_end.coerce_value(const.value, const.type)
        self.operations.append(Operation(OpType.PUSH_VALUE, value))
        self.curr_type = dtype

    def vec3(self, vec: ast_defs.Vec3):
        if all(map(lambda x: isinstance(x, ast_defs.Constant), [vec.x, vec.y, vec.z])):
            self.operations.append(Operation(OpType.PUSH_VALUE, [
                self.back_end.convert(vec.x.value, vec.x.type, DataType.FLOAT),
                self.back_end.convert(vec.y.value, vec.y.type, DataType.FLOAT),
                self.back_end.convert(vec.z.value, vec.z.type, DataType.FLOAT),
            ]))
            self.curr_type = DataType.VEC3
            return
        self.compile_expr(vec.x)
        x_type = self.curr_type
        x_loc = len(self.operations) - 1
        self.compile_expr(vec.y)
        y_type = self.curr_type
        y_loc = len(self.operations) - 1
        self.compile_expr(vec.z)
        z_type = self.curr_type
        z_loc = len(self.operations) - 1
        self.compile_function('vec3', [x_type, y_type, z_type], [
                              x_loc, y_loc, z_loc])

    def name(self, name: ast_defs.Name):
        # We should only end up here when we want to 'load' a variable.
        # If the name doesn't exist yet, create a default value
        if not name.id in self.vars:
            dtype = self.back_end.create_input(
                self.operations, name.id, None, DataType.DEFAULT)
            self.vars[name.id] = dtype
        self.operations.append(Operation(OpType.GET_VAR, name.id))
        self.curr_type = self.vars[name.id]


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
            compiler = Compiler(GeometryNodesBackEnd())
            try:
                success = compiler.compile(f.read())
                print(GREEN + 'No internal errors' + ENDC)
                if verbose > 0:
                    print(
                        f'{YELLOW}Syntax errors{ENDC}' if not success else f'{BLUE}No syntax errors{ENDC}')
                if verbose > 1 and success:
                    print(compiler.errors)
                if verbose > 2:
                    print(*compiler.operations, sep='\n')
                num_passed += 1
            except NotImplementedError:
                print(RED + 'Internal errors' + ENDC)
                # if verbose > 0:
                #     print(
                #         f'{YELLOW}Syntax errors{ENDC}:' if compiler.parser.had_error else f'{BLUE}No syntax errors{ENDC}')
                # if verbose > 1 and compiler.parser.had_error:
                #     print(compiler.operations)
    print(f'Tests done: Passed: ({num_passed}/{tot_tests})')
