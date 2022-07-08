from math_formula.backends.main import BackEnd
from math_formula.parser import Parser, Error
from math_formula import ast_defs
from math_formula.backends.type_defs import *


class TypeChecker():
    def __init__(self, back_end: BackEnd) -> None:
        self.typed_repr: TyRepr = TyRepr(body=[])
        self.errors: list[Error] = []
        self.curr_node: ty_stmt = None
        self.back_end: BackEnd = back_end
        self.vars: dict[str, Var] = {}

    def error(self, msg: str, node: ast_defs.Ast):
        self.errors.append(Error(node.token, msg))

    def type_check(self, source: str) -> bool:
        parser = Parser(source)
        ast = parser.parse()
        self.errors = parser.errors
        if parser.had_error:
            return False
        statements = ast.body
        for statement in statements:
            if isinstance(statement, ast_defs.expr):
                self.check_expr(statement)
            elif isinstance(statement, ast_defs.FunctionDef):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.NodegroupDef):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.Out):
                raise NotImplementedError
            elif isinstance(statement, ast_defs.Assign):
                self.check_assign(statement)
            elif isinstance(statement, ast_defs.Loop):
                raise NotImplementedError
            else:
                assert False, "Unreachable code"
            self.typed_repr.body.append(self.curr_node)
            if self.errors != []:
                return False
        return True

    def assign_types(self, targets: list[Union[ast_defs.Name, None]], dtypes: list[DataType]) -> list[Union[Var, None]]:
        typed_targets = [None for _ in range(len(targets))]
        for i, target in enumerate(targets):
            if target is None:
                continue
            var = Var(StackType.SOCKET, [dtypes[i]], target.id, False)
            self.vars[target.id] = var
            typed_targets[i] = var
        return typed_targets

    def check_assign(self, assign: ast_defs.Assign):
        targets = assign.targets
        self.check_expr(assign.value)
        expr = self.curr_node
        assert isinstance(
            expr, ty_expr), 'Right hand side of assignment should be an expression'
        if expr.stype == StackType.EMPTY:
            return self.error(
                'Right hand side of assignment should resolve to a value', assign)
        elif len(targets) > 1 and expr.stype != StackType.STRUCT:
            if expr.dtype[0] == DataType.VEC3:
                if len(targets) > 3:
                    return self.error('Too many assignment targets.', assign)
                self.curr_node = TyAssign(
                    self.assign_types(
                        targets, [DataType.FLOAT for _ in range(3)]), expr)
            elif expr.dtype[0] == DataType.RGBA:
                if len(targets) > 4:
                    return self.error('Too many assignment targets.', assign)
                self.curr_node = TyAssign(
                    self.assign_types(
                        targets, [DataType.FLOAT for _ in range(4)]), expr)
            else:
                return self.error('Too many assignment targets.', assign)
            return
        # Assignment is fine, as long as there are more values than targets.
        if len(targets) > len(expr.dtype):
            return self.error('Too many assignment targets.', assign)
        self.curr_node = TyAssign(
            self.assign_types(targets, expr.dtype), expr)

    def check_expr(self, expr: ast_defs.expr):
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

    def resolve_function(self, name: str, args: list[ty_expr], ast: ast_defs.Ast):
        # TODO: Remove assumption that this is a built in node.
        node = dtype = None
        try:
            node, dtype = self.back_end.resolve_function(
                name, args)
        except TypeError as err:
            return self.error(err, ast)
        if dtype == []:
            stype = StackType.EMPTY
        elif len(dtype) == 1:
            stype = StackType.SOCKET
        else:
            stype = StackType.STRUCT
        self.curr_node = NodeCall(stype, dtype, node, args)

    def unary_op(self, un_op: ast_defs.UnaryOp):
        op = un_op.op
        self.check_expr(un_op.operand)
        expr = self.curr_node
        assert isinstance(
            expr, ty_expr), 'Argument to unary op should be an expression'

        if expr.stype == StackType.EMPTY:
            return self.error(un_op, 'Argument expression has no value.')
        if isinstance(op, ast_defs.Not):
            self.resolve_function('not', [expr], un_op)
        elif isinstance(op, ast_defs.USub):
            arg = Const(StackType.VALUE, [DataType.INT], -1)
            self.resolve_function('mul', [arg, expr], un_op)
        else:
            assert False, "Unreachable code"

    def bin_op(self, bin_op: ast_defs.BinOp):
        op = bin_op.op
        self.check_expr(bin_op.left)
        left = self.curr_node
        self.check_expr(bin_op.right)
        right = self.curr_node
        assert isinstance(left, ty_expr) and isinstance(
            right, ty_expr), 'Arguments to binop should be expressions'
        if left.stype == StackType.EMPTY or right.stype == StackType.EMPTY:
            return self.error(bin_op, 'Argument expression has no value.')
        if isinstance(op, ast_defs.And):
            self.resolve_function('and', [left, right], bin_op)
        elif isinstance(op, ast_defs.Or):
            self.resolve_function('or', [left, right], bin_op)
        elif isinstance(op, ast_defs.Add):
            self.resolve_function('add', [left, right], bin_op)
        elif isinstance(op, ast_defs.Div):
            self.resolve_function('div', [left, right], bin_op)
        elif isinstance(op, ast_defs.Mod):
            self.resolve_function('mod', [left, right], bin_op)
        elif isinstance(op, ast_defs.Mult):
            self.resolve_function('mul', [left, right], bin_op)
        elif isinstance(op, ast_defs.Pow):
            self.resolve_function('pow', [left, right], bin_op)
        elif isinstance(op, ast_defs.Sub):
            self.resolve_function('sub', [left, right], bin_op)
        elif isinstance(op, ast_defs.Eq):
            self.resolve_function('equal', [left, right], bin_op)
        elif isinstance(op, ast_defs.Gt):
            self.resolve_function('greater_than', [left, right], bin_op)
        elif isinstance(op, ast_defs.GtE):
            self.resolve_function('greater_equal', [left, right], bin_op)
        elif isinstance(op, ast_defs.Lt):
            self.resolve_function('less_than', [left, right], bin_op)
        elif isinstance(op, ast_defs.LtE):
            self.resolve_function('less_equal', [left, right], bin_op)
        elif isinstance(op, ast_defs.NotEq):
            self.resolve_function('not_equal', [left, right], bin_op)
        else:
            assert False, "Unreachable code"

    def constant(self, const: ast_defs.Constant):
        try:
            value, dtype = self.back_end.coerce_value(const.value, const.type)
        except TypeError as err:
            return self.error(err, const)
        self.curr_node = Const(StackType.VALUE, [dtype], value)

    def vec3(self, vec: ast_defs.Vec3):
        if all(map(lambda x: isinstance(x, ast_defs.Constant), [vec.x, vec.y, vec.z])):
            self.curr_node = Const(StackType.VALUE, [DataType.VEC3], [
                self.back_end.convert(vec.x.value, vec.x.type, DataType.FLOAT),
                self.back_end.convert(vec.y.value, vec.y.type, DataType.FLOAT),
                self.back_end.convert(vec.z.value, vec.z.type, DataType.FLOAT),
            ])
            return
        # At least one of the argument is not a constant, so we need a combine XYZ node.
        self.check_expr(vec.x)
        x = self.curr_node
        self.check_expr(vec.y)
        y = self.curr_node
        self.check_expr(vec.z)
        z = self.curr_node
        assert isinstance(x, ty_expr) and isinstance(y, ty_expr) and isinstance(
            z, ty_expr), 'Arguments to combine XYZ should be expressions'
        if x.stype == StackType.EMPTY or y.stype == StackType.EMPTY or z.stype == StackType.EMPTY:
            return self.error('Argument expression has no value', vec)
        self.resolve_function('vec3', [x, y, z], vec)

    def name(self, name: ast_defs.Name):
        # We should only end up here when we want to 'load' a variable.
        # If the variable doesn't exist yet, create an empty
        if not name.id in self.vars:
            self.vars[name.id] = Var(
                StackType.SOCKET, [DataType.UNKNOWN], name.id, needs_instantion=True)
        self.curr_node = self.vars[name.id]


if __name__ == '__main__':
    import os
    from math_formula.backends.geometry_nodes import GeometryNodesBackEnd
    add_on_dir = os.path.dirname(
        os.path.realpath(__file__))
    test_directory = os.path.join(add_on_dir, 'tests')
    filenames = os.listdir(test_directory)
    verbose = 2
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
            type_checker = TypeChecker(GeometryNodesBackEnd())
            try:
                failed = not type_checker.type_check(f.read())
                print(GREEN + 'No internal errors' + ENDC)
                if verbose > 0:
                    print(
                        f'{YELLOW}Syntax errors{ENDC}' if failed else f'{BLUE}No syntax errors{ENDC}')
                if verbose > 1 and failed:
                    print(type_checker.errors)
                if verbose > 2:
                    print(ast_defs.dump(
                        type_checker.typed_repr, ty_ast, indent='.'))
                num_passed += 1
            except NotImplementedError:
                print(RED + 'Internal errors' + ENDC)
    print(f'Tests done: Passed: ({num_passed}/{tot_tests})')
