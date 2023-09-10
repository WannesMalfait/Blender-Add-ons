from copy import copy

from . import ast_defs
from .backends import type_defs as td
from .backends.main import BackEnd
from .mf_parser import Error, Parser


class TypeChecker:
    def __init__(
        self, back_end: BackEnd, functions: dict[str, list[td.TyFunction]] = {}
    ) -> None:
        self.typed_repr: td.TyRepr = td.TyRepr(body=[])
        self.errors: list[Error] = []
        self.curr_node: td.ty_stmt | None = None
        self.back_end: BackEnd = back_end
        self.vars: dict[str, td.Var] = {}
        # Can have multiple functions with same name, but different
        # type signatures.
        self.functions: dict[str, list[td.TyFunction]] = functions
        # Only set when inside a function definition
        self.function_outputs: list[td.TyArg] = []
        self.used_function_outputs: list[bool] = []

    def error(self, msg: str, node: ast_defs.Ast):
        self.errors.append(Error(node.token, msg))
        raise TypeError

    def type_check(self, source: str) -> bool:
        parser = Parser(source)
        ast = parser.parse()
        self.errors = parser.errors
        if parser.had_error:
            return False
        statements = ast.body
        for statement in statements:
            try:
                self.check_statement(statement)
            except TypeError:
                return False
            if self.curr_node is not None:
                # Could be None for function definition
                self.typed_repr.body.append(self.curr_node)
        return True

    def check_statement(self, stmt: ast_defs.stmt, in_function=False):
        if isinstance(stmt, ast_defs.expr):
            self.check_expr(stmt)
        elif isinstance(stmt, ast_defs.Assign):
            self.check_assign(stmt)
        elif isinstance(stmt, ast_defs.Loop):
            raise NotImplementedError
        elif isinstance(stmt, ast_defs.FunctionDef):
            if in_function:
                return self.error(
                    "No function definitions inside a function allowed", stmt
                )
            self.check_function_def(stmt)
        elif isinstance(stmt, ast_defs.NodegroupDef):
            if in_function:
                return self.error(
                    "No node group definitions inside a function allowed", stmt
                )
            self.check_function_def(stmt)
        elif isinstance(stmt, ast_defs.Out):
            if not in_function:
                return self.error("Out statements only allowed inside functions", stmt)
            self.check_out(stmt)
        else:
            assert False, "Unreachable code"

    def out_types(
        self,
        targets: list[td.TyArg | None],
        dtypes: list[td.DataType],
        ast_targets: list[None | ast_defs.Name],
    ):
        for target, dtype, ast_target in zip(targets, dtypes, ast_targets):
            if target is None:
                continue
            if not self.back_end.can_convert(dtype, target.dtype):
                if ast_target is None:
                    return
                return self.error(
                    f"Can't assign value of type {dtype._name_} to output of type {target.dtype._name_}",
                    ast_target,
                )

    def check_out(self, out_stmt: ast_defs.Out):
        # First check if all the target names are actually the output names.
        out_names = [out.name for out in self.function_outputs]
        out_targets: list[td.TyArg | None] = []
        target_indices: list[int | None] = []
        for target in out_stmt.targets:
            if target is None:
                target_indices.append(None)
                out_targets.append(None)
                continue
            if target.id not in out_names:
                return self.error(
                    f'Function output target "{target.id}" doesn\'t match one of the functions output names.',
                    target,
                )
            index = out_names.index(target.id)
            self.used_function_outputs[index] = True
            target_indices.append(index)
            out_targets.append(self.function_outputs[index])
        self.check_expr(out_stmt.value)
        expr = self.curr_node
        assert isinstance(
            expr, td.ty_expr
        ), "Right hand side of assignment should be an expression"
        if expr.stype == td.StackType.EMPTY:
            return self.error(
                "Right hand side of assignment should resolve to a value", out_stmt
            )
        elif len(out_targets) > 1 and expr.stype != td.StackType.STRUCT:
            if expr.dtype[0] == td.DataType.VEC3:
                if len(out_targets) > 3:
                    return self.error("Too many assignment targets.", out_stmt)
                self.out_types(
                    out_targets, [td.DataType.FLOAT for _ in range(3)], out_stmt.targets
                )
            elif expr.dtype[0] == td.DataType.RGBA:
                if len(out_targets) > 4:
                    return self.error("Too many assignment targets.", out_stmt)
                self.out_types(
                    out_targets, [td.DataType.FLOAT for _ in range(4)], out_stmt.targets
                )
            else:
                return self.error("Too many assignment targets.", out_stmt)
            return
        # Assignment is fine, as long as there are more values than targets.
        if len(out_targets) > len(expr.dtype):
            return self.error("Too many assignment targets.", out_stmt)
        self.out_types(out_targets, expr.dtype, out_stmt.targets)
        self.curr_node = td.TyOut(target_indices, expr)

    def check_arg(self, arg: ast_defs.arg) -> td.Union[None, td.ValueType]:
        if arg.default is None:
            return None
        self.check_expr(arg.default)
        default_value = self.curr_node
        if not isinstance(default_value, td.Const):
            return self.error(
                "Default value should be a value not an expression.", arg.default
            )
        try:
            return self.back_end.convert(
                default_value.value, default_value.dtype[0], arg.type
            )
        except:
            # TODO: Only except the relevant errors.
            return self.error(
                f"Can't convert {default_value} to value of type {arg.type._name_}",
                arg.default,
            )

    def check_function_def(
        self, fun_def: td.Union[ast_defs.FunctionDef, ast_defs.NodegroupDef]
    ):
        inputs = []
        outputs = []
        for arg in fun_def.args:
            inputs.append(td.TyArg(arg.arg, arg.type, self.check_arg(arg)))
        for ret in fun_def.returns:
            outputs.append(td.TyArg(ret.arg, ret.type, self.check_arg(ret)))
        outer_vars = self.vars
        self.vars = {}
        for arg in fun_def.args:
            var = td.Var(td.StackType.SOCKET, [arg.type], [], arg.arg, False)
            self.vars[arg.arg] = var
        body = []
        self.function_outputs = outputs
        self.used_function_outputs = [False for _ in range(len(outputs))]
        for stmt in fun_def.body:
            self.check_statement(stmt, in_function=True)
            checked_stmt = self.curr_node
            assert checked_stmt is not None, "There should be a statement"
            body.append(checked_stmt)
        is_nodegroup = isinstance(fun_def, ast_defs.NodegroupDef)
        if fun_def.name in self.functions:
            # Insert at the start, because newer definitions have higher priority
            self.functions[fun_def.name].insert(
                0,
                td.TyFunction(
                    inputs,
                    outputs,
                    body,
                    self.used_function_outputs,
                    is_nodegroup,
                    fun_def.name,
                ),
            )
        else:
            self.functions[fun_def.name] = [
                td.TyFunction(
                    inputs,
                    outputs,
                    body,
                    self.used_function_outputs,
                    is_nodegroup,
                    fun_def.name,
                )
            ]
        self.vars = outer_vars
        self.function_outputs = []
        self.curr_node = None

    def assign_types(
        self, targets: list[td.Union[ast_defs.Name, None]], dtypes: list[td.DataType]
    ) -> list[td.Var | None]:
        typed_targets: list[td.Var | None] = [None for _ in range(len(targets))]
        for i, target in enumerate(targets):
            if target is None:
                continue
            var = td.Var(td.StackType.SOCKET, [dtypes[i]], [], target.id, False)
            self.vars[target.id] = var
            typed_targets[i] = var
        return typed_targets

    def check_assign(self, assign: ast_defs.Assign):
        targets = assign.targets
        self.check_expr(assign.value)
        expr = self.curr_node
        assert isinstance(
            expr, td.ty_expr
        ), "Right hand side of assignment should be an expression"
        if expr.stype == td.StackType.EMPTY:
            return self.error(
                "Right hand side of assignment should resolve to a value", assign
            )
        elif len(targets) > 1 and expr.stype != td.StackType.STRUCT:
            if expr.dtype[0] == td.DataType.VEC3:
                if len(targets) > 3:
                    return self.error("Too many assignment targets.", assign)
                self.curr_node = td.TyAssign(
                    self.assign_types(targets, [td.DataType.FLOAT for _ in range(3)]),
                    expr,
                )
            elif expr.dtype[0] == td.DataType.RGBA:
                if len(targets) > 4:
                    return self.error("Too many assignment targets.", assign)
                self.curr_node = td.TyAssign(
                    self.assign_types(targets, [td.DataType.FLOAT for _ in range(4)]),
                    expr,
                )
            else:
                return self.error("Too many assignment targets.", assign)
            return
        elif len(targets) == 1 and expr.stype == td.StackType.STRUCT:
            # Assign the whole struct to the target.
            if (target := targets[0]) is not None:
                var = td.Var(
                    td.StackType.STRUCT, expr.dtype, expr.out_names, target.id, False
                )
                self.vars[target.id] = var
                self.curr_node = td.TyAssign([var], expr)
            return
        # Assignment is fine, as long as there are more values than targets.
        if len(targets) > len(expr.dtype):
            return self.error("Too many assignment targets.", assign)
        self.curr_node = td.TyAssign(self.assign_types(targets, expr.dtype), expr)

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
            self.attribute(expr)
        elif isinstance(expr, ast_defs.Keyword):
            assert False, (
                "Unreachable: keyword arguments are handled when type checking"
                + "function calls"
            )
        elif isinstance(expr, ast_defs.Call):
            self.func_call(expr)
        else:
            print(expr, type(expr))
            assert False, "Unreachable code"

    def resolve_function(
        self,
        name: str,
        pos_args: list[td.ty_expr],
        keyword_args: list[tuple[str, td.ty_expr]],
        ast: ast_defs.Ast,
    ):
        try:
            func, dtype, out_names, keyword_indices = self.back_end.resolve_function(
                name, pos_args, keyword_args, self.functions
            )
        except TypeError as err:
            return self.error(str(err), ast)
        if dtype == []:
            stype = td.StackType.EMPTY
        elif len(dtype) == 1:
            stype = td.StackType.SOCKET
            if dtype[0] == td.DataType.VEC3:
                out_names = ["x", "y", "z"]
            elif dtype[0] == td.DataType.RGBA:
                out_names = ["r", "g", "b", "a"]
        else:
            stype = td.StackType.STRUCT

        # Fill with "default" arguments first.
        final_args = pos_args + [
            td.Const(td.StackType.VALUE, [td.DataType.DEFAULT], [], None)
            for _ in range(len(func.inputs) - len(pos_args))
        ]
        for kw_i, input_i in enumerate(keyword_indices):
            final_args[input_i] = keyword_args[kw_i][1]

        if isinstance(func, td.TyFunction):
            self.curr_node = td.FunctionCall(stype, dtype, out_names, func, final_args)
        else:
            self.curr_node = td.NodeCall(stype, dtype, out_names, func, final_args)

    def func_call(self, call: ast_defs.Call):
        function_name = ""
        if isinstance(call.func, ast_defs.Attribute):
            function_name = call.func.attr
            # Add the implicit argument
            call.pos_args.insert(0, call.func.value)
        else:
            function_name = call.func.id
        ty_keyword_args = []
        if call.keyword_args != []:
            for keyword_arg in call.keyword_args:
                self.check_expr(keyword_arg.value)
                checked_arg = self.curr_node
                assert isinstance(
                    checked_arg, td.ty_expr
                ), "Argument should be an expression"
                ty_keyword_args.append((keyword_arg.arg, checked_arg))
        ty_args = []
        for pos_arg in call.pos_args:
            self.check_expr(pos_arg)
            checked_arg = self.curr_node
            assert isinstance(
                checked_arg, td.ty_expr
            ), "Argument should be an expression"
            ty_args.append(checked_arg)
        self.resolve_function(function_name, ty_args, ty_keyword_args, call)

    def unary_op(self, un_op: ast_defs.UnaryOp):
        op = un_op.op
        self.check_expr(un_op.operand)
        expr = self.curr_node
        assert isinstance(
            expr, td.ty_expr
        ), "Argument to unary op should be an expression"

        if expr.stype == td.StackType.EMPTY:
            return self.error("Argument expression has no value.", un_op)
        if isinstance(op, ast_defs.Not):
            self.resolve_function("_not", [expr], [], un_op)
        elif isinstance(op, ast_defs.USub):
            if isinstance(expr, td.Const) and (
                expr.dtype[0] == td.DataType.FLOAT or expr.dtype[0] == td.DataType.INT
            ):
                assert len(expr.dtype) == 1, "Should just be a float or an integer"
                assert isinstance(expr.value, (int, float)), "Checked above"
                expr.value *= -1
                return
            arg = td.Const(td.StackType.VALUE, [td.DataType.INT], [], -1)
            self.resolve_function("mul", [arg, expr], [], un_op)
        else:
            assert False, "Unreachable code"

    def bin_op(self, bin_op: ast_defs.BinOp):
        op = bin_op.op
        self.check_expr(bin_op.left)
        left = self.curr_node
        self.check_expr(bin_op.right)
        right = self.curr_node
        assert isinstance(left, td.ty_expr) and isinstance(
            right, td.ty_expr
        ), "Arguments to binop should be expressions"
        if left.stype == td.StackType.EMPTY or right.stype == td.StackType.EMPTY:
            return self.error("Argument expression has no value.", bin_op)
        if isinstance(op, ast_defs.And):
            self.resolve_function("_and", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Or):
            self.resolve_function("_or", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Add):
            self.resolve_function("add", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Div):
            self.resolve_function("div", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Mod):
            self.resolve_function("mod", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Mult):
            self.resolve_function("mul", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Pow):
            self.resolve_function("pow", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Sub):
            self.resolve_function("sub", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Eq):
            self.resolve_function("equal", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Gt):
            self.resolve_function("greater_than", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.GtE):
            self.resolve_function("greater_equal", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.Lt):
            self.resolve_function("less_than", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.LtE):
            self.resolve_function("less_equal", [left, right], [], bin_op)
        elif isinstance(op, ast_defs.NotEq):
            self.resolve_function("not_equal", [left, right], [], bin_op)
        else:
            assert False, "Unreachable code"

    def constant(self, const: ast_defs.Constant):
        value = dtype = None
        try:
            value, dtype = self.back_end.coerce_value(const.value, const.type)
        except TypeError as err:
            return self.error(str(err), const)
        self.curr_node = td.Const(td.StackType.VALUE, [dtype], [], value)

    def vec3(self, vec: ast_defs.Vec3):
        if (
            isinstance(vec.x, ast_defs.Constant)
            and isinstance(vec.y, ast_defs.Constant)
            and isinstance(vec.z, ast_defs.Constant)
        ):
            self.curr_node = td.Const(
                td.StackType.VALUE,
                [td.DataType.VEC3],
                ["x", "y", "z"],
                [
                    self.back_end.convert(vec.x.value, vec.x.type, td.DataType.FLOAT),
                    self.back_end.convert(vec.y.value, vec.y.type, td.DataType.FLOAT),
                    self.back_end.convert(vec.z.value, vec.z.type, td.DataType.FLOAT),
                ],
            )
            return
        # At least one of the arguments is not a constant, so we need a combine XYZ node.
        self.check_expr(vec.x)
        x = self.curr_node
        self.check_expr(vec.y)
        y = self.curr_node
        self.check_expr(vec.z)
        z = self.curr_node
        assert (
            isinstance(x, td.ty_expr)
            and isinstance(y, td.ty_expr)
            and isinstance(z, td.ty_expr)
        ), "Arguments to combine XYZ should be expressions"
        if (
            x.stype == td.StackType.EMPTY
            or y.stype == td.StackType.EMPTY
            or z.stype == td.StackType.EMPTY
        ):
            return self.error("Argument expression has no value", vec)
        self.resolve_function("vec3", [x, y, z], [], vec)

    def name(self, name: ast_defs.Name):
        # We should only end up here when we want to 'load' a variable.
        # If the variable doesn't exist yet, create an empty
        if name.id not in self.vars:
            var = td.Var(
                td.StackType.SOCKET,
                [td.DataType.UNKNOWN],
                [],
                name.id,
                needs_instantion=True,
            )
            self.vars[name.id] = var
            self.curr_node = var
            return
        var = self.vars[name.id]
        if var.needs_instantion:
            # At this point it doesn't need it anymore
            var = copy(var)
            var.needs_instantion = False
            self.vars[name.id] = var
        self.curr_node = var

    def attribute(self, attr: ast_defs.Attribute):
        self.check_expr(attr.value)
        expr = self.curr_node
        if not isinstance(expr, td.ty_expr) or expr.stype == td.StackType.EMPTY:
            self.error("Expected some value to retrieve attribute from.", attr)
        assert isinstance(expr, td.ty_expr), "Checked above"
        # See if the name is one of the outputs
        if attr.attr not in expr.out_names:
            return self.error(
                f'"{attr.attr}" does not match one of the output names: {expr.out_names}',
                attr,
            )
        if expr.stype == td.StackType.SOCKET:
            if expr.dtype[0] == td.DataType.VEC3:
                # Need to add a separate XYZ node for this to work.
                self.resolve_function("sep_xyz", [expr], [], attr)
                expr = self.curr_node
                assert isinstance(
                    expr, td.ty_expr
                ), "Result of sep_xyz should be an expression"
            elif expr.dtype[0] == td.DataType.RGBA:
                raise NotImplementedError
        index = expr.out_names.index(attr.attr)
        dtype = expr.dtype[index]
        out_names = []
        if dtype == td.DataType.VEC3:
            out_names = ["x", "y", "z"]
        elif dtype == td.DataType.RGBA:
            out_names = ["r", "g", "b", "a"]
        self.curr_node = td.GetOutput(
            td.StackType.SOCKET, [dtype], out_names, expr, index
        )


if __name__ == "__main__":
    import os

    from .backends.geometry_nodes import GeometryNodesBackEnd

    add_on_dir = os.path.dirname(os.path.realpath(__file__))
    test_directory = os.path.join(add_on_dir, "tests")
    filenames = os.listdir(test_directory)
    verbose = 3
    num_passed = 0
    tot_tests = 0
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[96m"
    ENDC = "\033[0m"
    for filename in filenames:
        # if filename != 'functions':
        #     continue
        tot_tests += 1
        print(f"Testing: {BOLD}{filename}{ENDC}:  ", end="")
        with open(os.path.join(test_directory, filename), "r") as f:
            type_checker = TypeChecker(GeometryNodesBackEnd())
            try:
                failed = not type_checker.type_check(f.read())
                print(GREEN + "No internal errors" + ENDC)
                if verbose > 0:
                    print(
                        f"{YELLOW}td.Type errors{ENDC}"
                        if failed
                        else f"{BLUE}No type errors{ENDC}"
                    )
                if verbose > 1 and failed:
                    print(type_checker.errors)
                if verbose > 2:
                    print(ast_defs.dump(type_checker.typed_repr, td.ty_ast, indent="."))
                num_passed += 1
            except NotImplementedError:
                print(RED + "Internal errors" + ENDC)
    print(f"Tests done: Passed: ({num_passed}/{tot_tests})")
