from enum import IntEnum, auto
from typing import Union
from bpy.types import NodeSocket
from .parser import Parser, Error, Token, TokenType
from .nodes import functions as function_nodes
from .nodes import geometry as geometry_nodes
from .nodes import shading as shader_nodes
from .nodes.base import DataType, Socket, Value, ValueType, data_type_to_string, NodeFunction

MacroType = dict[str, list[Token]]


class InstructionType(IntEnum):
    VALUE = 0
    VAR = auto()
    GET_VAR = auto()
    GET_OUTPUT = auto()
    PROPERTIES = auto()
    FUNCTION = auto()
    # NODEGROUP = auto()
    # DEF_FUNCTION = auto()
    # DEF_NODEGROUP = auto()
    END_OF_STATEMENT = auto()
    ASSIGNMENT = auto()
    IGNORE = auto()


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
    # Swap the last 2 elements of the stack.
    SWAP_2 = auto()
    # Call the given function, all the arguments are on the stack. Push the output
    # onto the stack
    CALL_FUNCTION = auto()
    # Create an input node for the given type. The value is stack.pop().
    CREATE_INPUT = auto()
    # Clear the stack.
    END_OF_STATEMENT = auto()


class Instruction():
    def __init__(self, instruction: InstructionType, data, token: Token) -> None:
        """ Create an instruction of the given type. `data` should
        contain the information needed to execute the type of instruction,
        not the arguments. """
        self.instruction = instruction
        self.data = data
        self.token = token

    def __str__(self) -> str:
        return f'[{self.instruction.name}, {self.data}]'

    def __repr__(self) -> str:
        return self.__str__()


class Operation():
    def __init__(self, op_type: OpType,
                 data: Union[ValueType, NodeFunction]) -> None:
        self.op_type = op_type
        self.data = data

    def __str__(self) -> str:
        return f"({self.op_type.name}, {self.data})"

    def __repr__(self) -> str:
        return self.__str__()


class Variable():
    def __init__(self, name: str,
                 value: Value) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class TypeCheckValue():
    """
    Values stored on the simulation stack when type checking
    - Token is used to report errors
    - Value is used to determine types
    - Intermediate is used to tell if it came from an instruction or is
        an intermediate value.
    - AdjustTarget: If it's not intermediate then this gives the index in
        `checked_program` that came from this instruction.
    """

    def __init__(self, token: Token, value: Value, intermediate: bool, adjust_target: int) -> None:
        self.token = token
        self.value = value
        self.intermediate = intermediate
        self.adjust_target = adjust_target


class TypeCheckStruct():
    def __init__(self, token: Token, types: list[Socket]) -> None:
        self.token = token
        self.types = types


class TypeCheckVar():
    def __init__(self, token: Token, var: Variable, target: int) -> None:
        self.token = token
        self.var = var
        self.target = target


class TypeChecker():
    def __init__(self) -> None:
        self.value_stack: list[Union[TypeCheckValue, TypeCheckStruct]] = []
        self.var_stack: list[Union[TypeCheckVar, None]] = []
        self.vars: dict[str, DataType] = {}
        self.checked_program: list[Operation] = []
        self.errors: list[Error] = []

    def error_at(self, token: Token, msg: str) -> None:
        error = f'line:{token.line}:{token.col}: Error'
        expanded = False
        macro_token = token
        while macro_token.expanded_from is not None:
            macro_token = macro_token.expanded_from
            expanded = True
        if expanded:
            error = f'line:{macro_token.line}:{macro_token.col}: Error'
            error += f' at "{token.lexeme}" in macro "{macro_token.lexeme}":'
        else:
            error += f' at "{macro_token.lexeme}":'
        self.errors.append(Error(token, f'{error} {msg}'))

    def add_operation(self, op_type: OpType, data: Union[ValueType, NodeFunction]):
        self.checked_program.append(Operation(op_type, data))

    def test_conversion_of_args(self, arg_values: list[TypeCheckValue], expected_arguments: list[Socket]) -> bool:
        for i, arg_value in enumerate(arg_values):
            assert isinstance(arg_value.value, Value), 'Type checker bug'
            given = arg_value.value.data_type
            expected = expected_arguments[i].sock_type
            if given == DataType.DEFAULT or expected == given:
                continue
            elif given.can_convert(expected):
                if arg_value.intermediate:
                    continue
                else:
                    operation = self.checked_program[arg_value.adjust_target]
                    assert operation.op_type == OpType.PUSH_VALUE, 'Type checker bug'
                    operation.data = Value(
                        given, operation.data).convert(expected)
            else:
                self.error_at(arg_value.token,
                              f'Invalid type, can\'t convert from {data_type_to_string[given]} to {data_type_to_string[expected]}.')
                return False
        return True

    def multiply_overloading(self, function: NodeFunction, arg_count: int) -> tuple[list[TypeCheckValue], NodeFunction]:
        if arg_count == 0:
            return [], function
        elif arg_count == 1:
            arg = self.value_stack.pop()
            if arg.value.data_type in (DataType.VEC3, DataType.RGBA):
                function = function_nodes.VectorMath(['MULTIPLY'])
            return [arg], function
        arg2 = self.value_stack.pop()
        arg1 = self.value_stack.pop()
        if arg1.value.data_type in (DataType.VEC3, DataType.RGBA):
            if arg2.value.data_type in (DataType.VEC3, DataType.RGBA):
                function = function_nodes.VectorMath(['MULTIPLY'])
            else:
                function = function_nodes.VectorMath(['SCALE'])
        elif arg2.value.data_type in (DataType.VEC3, DataType.RGBA):
            function = function_nodes.VectorMath(['SCALE'])
            # Vector should be first input!
            self.add_operation(OpType.SWAP_2, None)
            arg1, arg2 = arg2, arg1
        arg_types = [arg1, arg2]
        return arg_types, function

    def other_overloading(self, function: NodeFunction, arg_count: int) -> tuple[list[TypeCheckValue], NodeFunction]:
        operator_name = function_nodes.Math._overloadable[function.prop_values[0][1]]
        arg_types = self.value_stack[-arg_count:]
        self.value_stack = self.value_stack[:-arg_count]
        vector_math = False
        for arg in arg_types:
            if arg.value.data_type in (DataType.VEC3, DataType.RGBA):
                vector_math = True
                break
        if vector_math:
            function = function_nodes.VectorMath([operator_name])
        return arg_types, function

    def type_check_arguments(self, function: NodeFunction, arg_count: int, token: Token) -> tuple[bool, NodeFunction]:
        expected_arguments = function.input_sockets()
        if len(expected_arguments) < arg_count:
            self.error_at(token,
                          f'Expected at most {len(expected_arguments)} argument(s), but got {arg_count} arguments')
            return False, function
        assert len(self.value_stack) >= arg_count, 'Bug in type checker'
        arg_types = None
        for v in self.value_stack[-arg_count:]:
            if isinstance(v, TypeCheckStruct):
                self.error_at(v.token, 'Function has more than 1 output.')
                return False, function
        if isinstance(function, function_nodes.Math):
            if function.prop_values[0][1] == 'MULTIPLY':
                arg_types, function = self.multiply_overloading(
                    function, arg_count)
                expected_arguments = function.input_sockets()

            elif function.prop_values[0][1] in function_nodes.Math._overloadable:
                arg_types, function = self.other_overloading(
                    function, arg_count)
                expected_arguments = function.input_sockets()
            else:
                arg_types = self.value_stack[-arg_count:]
                self.value_stack = self.value_stack[:-arg_count]
        else:
            arg_types = self.value_stack[-arg_count:]
            self.value_stack = self.value_stack[:-arg_count]
        if not self.test_conversion_of_args(arg_types, expected_arguments):
            return False, function
        if arg_count < len(expected_arguments):
            # Fill up with default values
            for _ in range(len(expected_arguments)-arg_count):
                # No need to type check these.
                self.add_operation(OpType.PUSH_VALUE, None)
        return True, function

    def type_check_function(self, instruction: Instruction) -> bool:
        """
        Check if the input arguments of the function are correct. If
        everything is fine then it is added to `checked_program`.
        The arguments needed to execute the function are checked and
        added to `checked_program` as well.
        """
        function, arg_count = instruction.data
        assert isinstance(function, NodeFunction)
        assert isinstance(arg_count, int)
        # Have to return `function` because modifying it inside the function
        # doesn't work.
        ok, function = self.type_check_arguments(
            function, arg_count, instruction.token)
        if not ok:
            return False
        self.add_operation(OpType.CALL_FUNCTION, function)
        outputs = function.output_sockets()
        if len(outputs) == 1:
            self.value_stack.append(TypeCheckValue(instruction.token, Value(
                outputs[0].sock_type, NodeSocket), True, 0))
        elif len(outputs) > 1:
            self.value_stack.append(
                TypeCheckStruct(instruction.token, outputs))
        return True

    def try_assign(self, var: TypeCheckVar, value: Value) -> bool:
        var_type = var.var.value.data_type
        value_type = value.data_type
        if value_type == DataType.DEFAULT:
            self.add_operation(OpType.CREATE_INPUT,
                               var_type)
        else:
            if var_type.can_convert(value_type):
                if value.value != NodeSocket:
                    self.add_operation(OpType.CREATE_INPUT, value_type)
            else:
                self.error_at(
                    var.token, f'Can\'t assign value with type {data_type_to_string[value_type]} to variable with type {data_type_to_string[var_type]}.')
                return False
        self.add_operation(OpType.CREATE_VAR, var.var.name)
        self.vars[var.var.name] = value_type
        return True

    def type_check_assignment(self, assignment_instruction: Instruction) -> bool:
        num_vars = assignment_instruction.data
        assert isinstance(
            num_vars, int), 'Bug in parser, wrong data for assignment'
        assert len(
            self.value_stack) == 1, 'Should always be one element (a Value or a Struct).'
        value = self.value_stack.pop()
        value_is_struct = isinstance(value, TypeCheckStruct)
        if num_vars > 1:
            if value_is_struct:
                if len(value.types) < num_vars:
                    self.error_at(assignment_instruction.token,
                                  f'Too many variables, expected at most {len(value.types)}.')
                    return False
            else:
                if value.value.data_type == DataType.VEC3 and 3 < num_vars:
                    self.error_at(assignment_instruction.token,
                                  'Too many variables, expected at most 3.')
                    return False
        vars = self.var_stack[-num_vars:]
        self.var_stack[:] = self.var_stack[:-num_vars]
        if value_is_struct:
            for i, var in enumerate(vars):
                if var is None:
                    continue
                self.add_operation(OpType.GET_OUTPUT, i)
                self.add_operation(OpType.CREATE_VAR, var.var.name)
                var_type = var.var.value.data_type
                value_type = value.types[i].sock_type
                if not var_type.can_convert(value_type):
                    self.error_at(
                        var.token, f'Can\'t assign value of type {data_type_to_string[value_type]} to variable of type {[data_type_to_string[var_type]]}')
                    return False
                self.vars[var.var.name] = value_type
        else:
            if num_vars > 1 and value.value.data_type == DataType.VEC3:
                self.add_operation(OpType.CALL_FUNCTION,
                                   function_nodes.SeparateXYZ([]))
                for i, var in enumerate(vars):
                    if var is None:
                        continue
                    self.add_operation(OpType.GET_OUTPUT, i)
                    self.add_operation(OpType.CREATE_VAR, var.var.name)
                    var_type = var.var.value.data_type
                    if not var_type.can_convert(DataType.FLOAT):
                        self.error_at(
                            var.token, f'Can\'t assign value of type {data_type_to_string[DataType.FLOAT]} to variable of type {[data_type_to_string[var_type]]}')
                        return False
                    self.vars[var.var.name] = DataType.FLOAT
            else:
                self.try_assign(vars[0], value.value)
                for var in vars[1:]:
                    if var is None:
                        continue
                    if value.value.value == NodeSocket:
                        self.add_operation(OpType.GET_VAR, vars[0].var.name)
                    else:
                        self.add_operation(
                            OpType.PUSH_VALUE, value.value.value)
                    self.try_assign(var, value.value)
        return True

    def type_check(self, program: list[Instruction]) -> bool:
        """
        Go through all the instructions and make sure types and argument counts match up.
        This will modify the program in some cases to ensure correctness.
        """
        assert InstructionType.IGNORE.value == 8, 'Exhaustive handling of instructions'
        ip = 0
        no_errors = True
        while ip < len(program):
            instruction = program[ip]
            itype = instruction.instruction
            panic = False
            if itype == InstructionType.VALUE:
                assert isinstance(instruction.data,
                                  Value), 'Parser Bug: non-value data.'
                self.add_operation(OpType.PUSH_VALUE, instruction.data.value)
                self.value_stack.append(TypeCheckValue(
                    instruction.token, instruction.data, False, len(self.checked_program) - 1))
            elif itype == InstructionType.VAR:
                assert isinstance(instruction.data, Variable), 'Parser bug.'
                self.var_stack.append(TypeCheckVar(
                    instruction.token, instruction.data, len(self.checked_program)-1))
            elif itype == InstructionType.GET_VAR:
                assert isinstance(
                    instruction.data, str), 'Parser Bug: variable name should be a str.'
                name = instruction.data
                if not name in self.vars:
                    self.error_at(instruction.token, 'Unknown variable name.')
                    panic = True
                else:
                    data_type = self.vars[name]
                    self.value_stack.append(TypeCheckValue(
                        instruction.token, Value(data_type, NodeSocket), True, 0))
                    self.add_operation(OpType.GET_VAR, name)
            elif itype == InstructionType.GET_OUTPUT:
                value = self.value_stack.pop()
                if isinstance(value, TypeCheckValue):
                    if value.value.data_type == DataType.VEC3:
                        function = function_nodes.SeparateXYZ([])
                        self.add_operation(
                            OpType.CALL_FUNCTION, function)
                        value = TypeCheckStruct(
                            None, function.output_sockets())
                    else:
                        self.error_at(instruction.token,
                                      f'Previous expression only had one output. HINT: remove ".{instruction.token.lexeme}".')
                        panic = True
                found = False
                for i, socket in enumerate(value.types):
                    if socket.name == instruction.data:
                        self.add_operation(OpType.GET_OUTPUT, i)
                        self.value_stack.append(TypeCheckValue(
                            instruction.token, Value(socket.sock_type, NodeSocket), True, 0))
                        found = True
                        break
                if not found:
                    self.error_at(
                        instruction.token, 'Output name does not match any of the output socket names.')
                    panic = True
            elif itype == InstructionType.PROPERTIES:
                assert False, 'Parser Bug: Properties should have been handled already.'
            elif itype == InstructionType.FUNCTION:
                if not self.type_check_function(instruction):
                    panic = True
            elif itype == InstructionType.END_OF_STATEMENT:
                self.value_stack = []
                self.var_stack = []
                self.add_operation(OpType.END_OF_STATEMENT, None)
            elif itype == InstructionType.ASSIGNMENT:
                if not self.type_check_assignment(instruction):
                    panic = True
            elif itype == InstructionType.IGNORE:
                self.var_stack.append(None)
            if panic:
                no_errors = False
                while ip + 1 < len(program) and program[ip+1].instruction != InstructionType.END_OF_STATEMENT:
                    ip += 1
            else:
                ip += 1
        return no_errors


class Compiler():
    def __init__(self) -> None:
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []
        self.checked_program: list[Operation] = []

    def compile(self, source: str, macros: MacroType, tree_type: str) -> bool:
        self.instructions = []
        parser = Parser(source, macros, tree_type)
        parser.advance()
        while not parser.match(TokenType.EOL):
            parser.declaration()
        parser.consume(TokenType.EOL, 'Expect end of expression.')
        self.instructions = parser.instructions
        self.errors = parser.errors
        if parser.had_error:
            return False
        type_checker = TypeChecker()
        result = type_checker.type_check(self.instructions)
        self.checked_program = type_checker.checked_program
        # TODO: Maybe keep these separate?
        self.errors += type_checker.errors
        return result

    @ staticmethod
    def get_tokens(source: str) -> list[Token]:
        tokens = []
        parser = Parser(source, {}, '')
        scanner = parser.scanner
        while(token := scanner.scan_token()).token_type != TokenType.EOL:
            tokens.append(token)
        return tokens


if __name__ == '__main__':
    tests = [
        'let x;let y=2+5**-.5; let z = -y*x;',
        'let a,_,c = "HEY";',
        'let a:float = 1;',
        'let _,b = 2;'
    ]
    compiler = Compiler()
    for test in tests:
        print('\nTESTING:', test)
        success = compiler.compile(test)
        print(f'Compilation {"success" if success else "failed"}\n')
        if not success:
            for error in compiler.errors:
                print(error.message)
        for token in compiler.instructions:
            print(token)
