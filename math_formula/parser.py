import math
from typing import Union
from bpy.types import NodeSocket, Operator
from .scanner import TokenType, Token, Scanner
from .nodes import functions as function_nodes
from .nodes.base import DataType, Socket, Value, ValueType, string_to_data_type, data_type_to_string, NodeFunction
from enum import IntEnum, auto


MacroType = dict[str, list[Token]]


class Precedence(IntEnum):
    NONE = 0
    ASSIGNMENT = auto()  # =
    COMPARISON = auto()  # < >
    TERM = auto()       # + -
    FACTOR = auto()     # * /
    UNARY = auto()      # -
    EXPONENT = auto()   # ^ **
    CALL = auto()       # . () {}
    PRIMARY = auto()


class InstructionType(IntEnum):
    VALUE = 0
    # STRUCT = auto()
    VAR = auto()
    GET_VAR = auto()
    PROPERTIES = auto()
    FUNCTION = auto()
    # NODEGROUP = auto()
    # DEF_FUNCTION = auto()
    # DEF_NODEGROUP = auto()
    END_OF_STATEMENT = auto()
    ASSIGNMENT = auto()
    IGNORE = auto()


class OpType(IntEnum):
    PUSH_VALUE = 0
    CALL_FUNCTION = auto()
    END_OF_STATEMENT = auto()


class Operation():
    def __init__(self, op_type: OpType,
                 data: Union[ValueType, NodeFunction]) -> None:
        self.op_type = op_type
        self.data = data


class Variable():
    def __init__(self, name: str,
                 value: Value) -> None:
        self.name = name
        self.value = value

    def __str__(self) -> str:
        return f"{self.name}: {self.value}"

    def __repr__(self) -> str:
        return self.__str__()


class Error():
    def __init__(self, token: Token, message: str) -> None:
        self.token = token
        self.message = message

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'Line: {self.token.line}: {self.message}'


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


class ParseRule():
    def __init__(self, prefix, infix, precedence: Precedence) -> None:
        self.prefix = prefix
        self.infix = infix
        self.precedence = precedence


class Parser():

    def __init__(self, source: str, macro_storage: MacroType) -> None:
        self.scanner = Scanner(source)
        self.token_buffer: list[Token] = []
        self.current: Token = None
        self.previous: Token = None
        self.had_error: bool = False
        self.panic_mode: bool = False
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []
        self.macro_storage = macro_storage

    def error_at_current(self, message: str) -> None:
        self.error_at(self.current, message)

    def error(self, message: str) -> None:
        self.error_at(self.previous, message)

    def error_at(self, token: Token, message: str) -> None:
        if self.panic_mode:
            return
        self.panic_mode = True
        error = 'Error'
        if token.token_type == TokenType.EOL:
            error += ' at end:'
        elif token.token_type == TokenType.ERROR:
            pass
        else:
            error += f' at "{token.lexeme}":'
            if token.expanded_from is not None:
                error += f' expanded from macro "{token.expanded_from.lexeme}": '
        # TODO: Better handling of errors
        self.errors.append(Error(token, f'{error} {message}'))
        self.had_error = True

    def consume(self, token_type: TokenType, message: str) -> None:
        if self.check(token_type):
            self.advance()
            return

        self.error_at_current(message)

    def add_tokens(self, tokens: list[Token]) -> None:
        first = tokens[0]
        self.token_buffer.append(self.current)
        # Need to reverse because we pop later
        self.token_buffer += list(reversed(tokens[1:]))

        self.current = first

    def advance(self) -> None:
        self.previous = self.current

        # Get tokens till not an error
        while True:
            if self.token_buffer != []:
                self.current = self.token_buffer.pop()
            else:
                self.current = self.scanner.scan_token()
            if self.current.token_type != TokenType.ERROR:
                break
            token, message = self.current.lexeme
            self.error_at_current(f': {message}: {token}')

    def get_rule(self, token_type: TokenType) -> ParseRule:
        return rules[token_type.value]

    def parse_precedence(self, precedence: Precedence) -> None:
        self.advance()
        prefix_rule = self.get_rule(self.previous.token_type).prefix
        if prefix_rule is None:
            self.error('Expect expression.')
            return
        can_assign = precedence.value <= Precedence.ASSIGNMENT.value
        prefix_rule(self, can_assign)
        while precedence.value <= self.get_rule(self.current.token_type).precedence.value:
            self.advance()
            infix_rule = self.get_rule(self.previous.token_type).infix
            infix_rule(self, can_assign)
        if can_assign and self.match(TokenType.EQUAL):
            self.error('Invalid assignment target.')

    def check(self, token_type: TokenType) -> bool:
        return self.current.token_type == token_type

    def match(self, token_type: TokenType) -> bool:
        if not self.check(token_type):
            return False
        self.advance()
        return True

    def expression(self) -> None:
        self.parse_precedence(Precedence.ASSIGNMENT)

    def statement(self) -> None:
        self.expression()
        # Get optional semicolon at end of expression
        self.match(TokenType.SEMICOLON)
        # self.consume(TokenType.SEMICOLON, 'Expect ";" after expression.')
        # A statement has no return value, this tells the operator
        # to remove remaining values from the stack.
        self.instructions.append(Instruction(
            InstructionType.END_OF_STATEMENT, None, self.previous))

    def parse_type(self) -> DataType:
        if self.match(TokenType.COLON):
            if self.match(TokenType.IDENTIFIER):
                if self.previous.lexeme in string_to_data_type:
                    return string_to_data_type[self.previous.lexeme]
                else:
                    self.error(f'Invalid data type: {self.previous.lexeme}.')
            else:
                self.error('Expected a data type')
        return DataType.UNKNOWN

    def parse_variable_and_type(self) -> Instruction:
        token = self.previous
        name = token.lexeme
        var_type = self.parse_type()

        return Instruction(InstructionType.VAR, Variable(name, Value(var_type, None)), token)

    def parse_variable(self, message: str) -> int:
        # Possible options after let:
        # - let x = ...;
        # - let a, b, c = ...;
        # - let a, _, b = ...;
        # Then after each variable there can be a colon (:)
        # followed by a type. So something like
        # let x: float = 3;
        num_vars = 0
        while not self.check(TokenType.EQUAL) or self.check(TokenType.SEMICOLON):
            if self.match(TokenType.IDENTIFIER):
                self.instructions.append(self.parse_variable_and_type())
                num_vars += 1
            elif self.match(TokenType.UNDERSCORE):
                self.instructions.append(
                    Instruction(InstructionType.IGNORE, None, self.previous))
                num_vars += 1
            else:
                self.error_at_current(message)
            if not self.match(TokenType.COMMA):
                break
        if num_vars == 0:
            self.error(message)
        return num_vars

    def add_value_instruction(self, value: Value) -> None:
        self.instructions.append(Instruction(
            InstructionType.VALUE, value, self.previous))

    def variable_declaration(self) -> None:
        let_token = self.previous
        num_vars = self.parse_variable(
            'Expect variable name or "_" after "let".')
        if self.match(TokenType.EQUAL):
            self.expression()
        else:
            # Something like 'let x: float;' gets de-sugared to 'let x = 0.0;'
            self.add_value_instruction(Value(DataType.DEFAULT, None))
        self.instructions.append(Instruction(
            InstructionType.ASSIGNMENT, num_vars, let_token))
        # Get optional semicolon at end of expression
        self.match(TokenType.SEMICOLON)
        self.instructions.append(Instruction(
            InstructionType.END_OF_STATEMENT, None, self.previous))

    def declaration(self) -> None:
        if self.match(TokenType.LET):
            self.variable_declaration()
        else:
            self.statement()

        if self.panic_mode:
            self.synchronize()

    def argument_list(self, closing_token: Token) -> int:
        arg_count = 0
        if not self.check(TokenType.RIGHT_PAREN):
            self.expression()
            arg_count += 1
            while self.match(TokenType.COMMA):
                self.expression()
                arg_count += 1
        self.consume(closing_token.token_type,
                     f'Expect "{closing_token.lexeme}" after arguments.')
        return arg_count

    def synchronize(self) -> None:
        self.panic_mode = False
        while self.previous.token_type != TokenType.EOL:
            if self.previous.token_type == TokenType.SEMICOLON:
                return
            if self.current.token_type == TokenType.LET:
                return
            self.advance()


def make_int(self: Parser, can_assign: bool) -> None:
    value = int(self.previous.lexeme)
    self.add_value_instruction(Value(DataType.INT, value))


def make_float(self: Parser, can_assign: bool) -> None:
    value = float(self.previous.lexeme)
    self.add_value_instruction(Value(DataType.FLOAT, value))


def python(self: Parser, can_assign: bool) -> None:
    expression = self.previous.lexeme[1:]
    value = 0
    try:
        value = eval(expression, vars(math))
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as err:
        self.error(f'Invalid python syntax: {err}.')
    self.add_value_instruction(Value(value, DataType.FLOAT))


def default(self: Parser, can_assign: bool) -> None:
    self.add_value_instruction(Value(DataType.DEFAULT, None))


def identifier(self: Parser, can_assign: bool) -> None:
    identifier_token = self.previous
    name = identifier_token.lexeme
    if name in self.macro_storage:
        rhs = self.macro_storage[name]
        for token in rhs:
            token.expanded_from = identifier_token
            token.times_expanded = self.previous.times_expanded + 1
            if token.times_expanded > 10:
                self.error_at(identifier_token,
                              'Expansion limit exceeded. Possible recursion')
        self.add_tokens(rhs)
        self.expression()
    else:
        self.instructions.append(Instruction(
            InstructionType.GET_VAR, name, identifier_token)
        )


def string(self: Parser, can_assign: bool) -> None:
    self.add_value_instruction(Value(
        # Get rid of the quotes surrounding it
        DataType.STRING, self.previous.lexeme[1:-1])
    )


def boolean(self: Parser, can_assign: bool) -> None:
    self.add_value_instruction(
        Value(DataType.BOOL, self.previous.lexeme == 'true'))


def grouping(self: Parser, can_assign: bool) -> None:
    self.expression()
    self.consume(TokenType.RIGHT_PAREN, 'Expect closing ")" after expression.')


def unary(self: Parser, can_assign: bool) -> None:
    operator_token = self.previous
    operator_type = operator_token.token_type
    # Compile the operand
    self.parse_precedence(Precedence.UNARY)

    if operator_type == TokenType.MINUS:
        # unary minus (-x) in shader nodes is x * -1
        # postfix: x -1 *
        self.add_value_instruction(Value(DataType.INT, -1))
        self.instructions.append(Instruction(
            InstructionType.FUNCTION, (function_nodes.Math(['MULTIPLY']), 2), operator_token))
    else:
        # Shouldn't happen
        assert False, "Unreachable code"


def make_vec3_from_values(args: list[Instruction]) -> list[float]:
    vec3 = []
    for arg in args:
        val = arg.data.value
        if val is None:
            vec3.append(0.0)
        elif isinstance(val, float):
            vec3.append(val)
        else:
            assert False, 'Unreachable code'
    return vec3


def make_vector(self: Parser, can_assign: bool) -> None:
    bracket_token = self.previous
    function = function_nodes.CombineXYZ([])
    arg_count = self.argument_list(Token('}', TokenType.RIGHT_BRACE))
    self.instructions.append(Instruction(
        InstructionType.FUNCTION, (function, arg_count), bracket_token))
    # # Check if the arguments are just regular values. In that case
    # # we don't need a combine XYZ node.
    # args = self.instructions[-4:-1]
    # no_expressions = True
    # for arg in args:
    #     if not isinstance(arg.data, Value):
    #         no_expressions = False
    #         break
    # if no_expressions:
    #     self.instructions = self.instructions[:-4]
    #     # Get rid of the result of CombineXYZ
    #     self.add_value_instruction(
    #         Value(DataType.VEC3, make_vec3_from_values(args)))


def call(self: Parser, can_assign: bool) -> None:
    # We need to get rid of this instruction, and replace it
    # with a function call.
    prev_instruction = self.instructions.pop()
    props = []
    if prev_instruction.instruction == InstructionType.PROPERTIES:
        props = [self.instructions.pop().data.value for _ in range(
            prev_instruction.data)]
        # This is the function name now
        prev_instruction = self.instructions.pop()
    if prev_instruction.instruction != InstructionType.GET_VAR:
        self.error('Expected callable object')
    func_name = prev_instruction.data
    function = None
    for dict in (function_nodes.functions,):
        if func_name in dict:
            function_cls: NodeFunction = dict[func_name]
            if len(props) > len(function_cls._props):
                self.error(
                    f'Too many properties for function {func_name}, expected at most {len(function_cls._props)}')
            else:
                invalid = function_cls.invalid_prop_values(props)
                if invalid != []:
                    error_list = ['_' if x is None else x for x in invalid]
                    func_props = [prop[0] for prop in function_cls._props]
                    self.error(
                        f'Invalid property values: {error_list}. HINT: {func_name} has the following properties: {func_props}.')
            function = function_cls(props)
            break
    if function is None:
        self.error(f'Unknown function {func_name}.')
        return
    arg_count = self.argument_list(Token(')', TokenType.RIGHT_PAREN))
    self.instructions.append(Instruction(
        InstructionType.FUNCTION, (function, arg_count), prev_instruction.token))


def properties(self: Parser, can_assign: bool) -> None:
    bracket_token = self.previous
    num_props = 0
    while not self.check(TokenType.RIGHT_SQUARE_BRACKET):
        if self.match(TokenType.STRING):
            string(self, can_assign)
            num_props += 1
        elif self.match(TokenType.UNDERSCORE):
            default(self, can_assign)
            num_props += 1
        else:
            self.error_at_current('Expect property values')
        if not self.match(TokenType.COMMA):
            break
    self.consume(TokenType.RIGHT_SQUARE_BRACKET, 'Expect closing "]".')
    self.instructions.append(Instruction(
        InstructionType.PROPERTIES, num_props, bracket_token))


def macro(self: Parser, can_assign: bool) -> None:
    # Simplified version of macros for now. No arguments.
    # We're turning something like this:
    # MACRO sub = math['SUBTRACT'];
    # into a dictionary entry with key 'sub' and value:
    # [Tokens] (list of tokens of the right hand side)
    self.consume(TokenType.IDENTIFIER, 'Expect macro name.')
    name = self.previous.lexeme
    self.consume(TokenType.EQUAL, 'Expect "=" after macro name.')
    rhs = []
    while not self.check(TokenType.SEMICOLON):
        if self.match(TokenType.MACRO):
            self.error('Definition of a macro inside of a macro.')
        self.advance()
        rhs.append(self.previous)
    self.macro_storage[name] = rhs


def separate(self: Parser, can_assign: bool) -> None:
    raise NotImplementedError


def binary(self: Parser, can_assign: bool) -> None:
    operator_token = self.previous
    operator_type = operator_token.token_type
    rule = self.get_rule(operator_type)
    self.parse_precedence(Precedence(rule.precedence.value + 1))

    # math: + - / * % > < **
    function = None
    if operator_type == TokenType.PLUS:
        function = function_nodes.Math(['ADD'])
    elif operator_type == TokenType.MINUS:
        function = function_nodes.Math(['SUBTRACT'])
    elif operator_type == TokenType.SLASH:
        function = function_nodes.Math(['DIVIDE'])
    elif operator_type == TokenType.STAR:
        function = function_nodes.Math(['MULTIPLY'])
    elif operator_type == TokenType.PERCENT:
        function = function_nodes.Math(['MODULO'])
    elif operator_type == TokenType.GREATER:
        function = function_nodes.Math(['GREATER_THAN'])
    elif operator_type == TokenType.LESS:
        function = function_nodes.Math(['LESS_THAN'])
    elif operator_type in (TokenType.STAR_STAR, TokenType.HAT):
        function = function_nodes.Math(['POWER'])
    else:
        assert False, "Unreachable code"
    self.instructions.append(Instruction(
        InstructionType.FUNCTION, (function, 2), operator_token))


rules: list[ParseRule] = [
    ParseRule(grouping, call, Precedence.CALL),  # LEFT_PAREN
    ParseRule(None, None, Precedence.NONE),  # RIGHT_PAREN
    ParseRule(None, properties, Precedence.CALL),  # LEFT_SQUARE_BRACKET
    ParseRule(None, None, Precedence.NONE),  # RIGHT_SQUARE_BRACKET
    ParseRule(make_vector, None, Precedence.NONE),  # LEFT_BRACE
    ParseRule(None, None, Precedence.NONE),  # RIGHT_BRACE
    ParseRule(None, None, Precedence.NONE),  # COMMA
    ParseRule(None, separate, Precedence.CALL),  # DOT
    ParseRule(None, None, Precedence.NONE),  # SEMICOLON
    ParseRule(None, None, Precedence.NONE),  # EQUAL
    ParseRule(unary, binary, Precedence.TERM),  # MINUS
    ParseRule(None, binary, Precedence.TERM),  # PLUS
    ParseRule(None, binary, Precedence.FACTOR),  # PERCENT
    ParseRule(None, binary, Precedence.FACTOR),  # SLASH
    ParseRule(None, binary, Precedence.EXPONENT),  # HAT
    ParseRule(None, binary, Precedence.COMPARISON),  # GREATER
    ParseRule(None, binary, Precedence.COMPARISON),  # LESS
    ParseRule(None, None, Precedence.NONE),  # DOLLAR
    ParseRule(None, None, Precedence.NONE),  # COLON
    ParseRule(default, None, Precedence.NONE),  # UNDERSCORE

    ParseRule(None, binary, Precedence.FACTOR),  # STAR
    ParseRule(None, binary, Precedence.EXPONENT),  # STAR_STAR
    ParseRule(None, None, Precedence.NONE),  # ARROW

    ParseRule(identifier, None, Precedence.NONE),  # IDENTIFIER
    ParseRule(make_int, None, Precedence.NONE),  # INT
    ParseRule(make_float, None, Precedence.NONE),  # FLOAT
    ParseRule(python, None, Precedence.NONE),  # PYTHON
    ParseRule(string, None, Precedence.NONE),  # STRING

    ParseRule(None, None, Precedence.NONE),  # LET
    ParseRule(None, None, Precedence.NONE),  # FUNCTION
    ParseRule(None, None, Precedence.NONE),  # NODEGROUP
    ParseRule(macro, None, Precedence.NONE),  # MACRO
    ParseRule(None, None, Precedence.NONE),  # SELF
    ParseRule(boolean, None, Precedence.NONE),  # TRUE
    ParseRule(boolean, None, Precedence.NONE),  # FALSE

    ParseRule(None, None, Precedence.NONE),  # ERROR
    ParseRule(None, None, Precedence.NONE),  # EOL
]

assert len(rules) == TokenType.EOL.value + 1, "Didn't handle all tokens!"


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


class TypeChecker():
    def __init__(self) -> None:
        self.stack: list[TypeCheckValue] = []
        self.vars = {}
        self.checked_program: list[Operation] = []
        self.errors: list[Error] = []

    def error_at(self, token: Token, msg: str) -> None:
        error = f'Error at "{token.lexeme}":'
        if token.expanded_from is not None:
            error += f' expanded from macro "{token.expanded_from.lexeme}": '
        self.errors.append(Error(token, f'{error} {msg}'))

    def add_operation(self, op_type: OpType, data: Union[ValueType, NodeFunction]):
        self.checked_program.append(Operation(op_type, data))

    # def type_check_operator(self, operator_name: str) -> None:
    #     arg2 = self.type_check_stack.pop()
    #     arg1 = self.type_check_stack.pop()
    #     function = None
    #     if arg1[0].data_type == DataType.VEC3 or arg2[0].data_type == DataType.VEC3:
    #         function = function_nodes.VectorMath([operator_name])
    #     else:
    #         function = function_nodes.Math([operator_name])
    #     self.test_conversion_of_args((arg1, arg2), function.input_sockets())
    #     self.type_check_outputs(function)

    # def type_check_multiply(self) -> None:
    #     arg2 = self.type_check_stack.pop()
    #     arg1 = self.type_check_stack.pop()
    #     function = None
    #     if arg1[0].data_type == DataType.VEC3:
    #         if arg2[0].data_type == DataType.VEC3:
    #             function = function_nodes.VectorMath(['MULTIPLY'])
    #         else:
    #             function = function_nodes.VectorMath(['SCALE'])
    #     elif arg2[0].data_type == DataType.VEC3:
    #         function = function_nodes.VectorMath(['SCALE'])
    #         # Vector should be first input!
    #         instruction1 = arg1[1]
    #         arg2, arg1 = arg1, arg2
    #         self.instructions.remove(instruction1)
    #         self.instructions.append(instruction1)
    #     else:
    #         function = function_nodes.Math(['MULTIPLY'])
    #     self.test_conversion_of_args((arg1, arg2), function.input_sockets())
    #     self.type_check_outputs(function)

    # def type_check_outputs(self, function: NodeFunction) -> None:
    #     self.instructions.append(Instruction(
    #         InstructionType.FUNCTION, function))
    #     outputs = function.output_sockets()
    #     if len(outputs) == 1:
    #         self.type_check_stack.append(
    #             (Value(outputs[0].sock_type, NodeSocket), self.instructions[-1]))
    #     else:
    #         raise NotImplementedError
    #         # for output in outputs:
    #         #     self.type_check_stack.append(Value(output.sock_type, NodeSocket))

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

    def type_check_arguments(self, function: NodeFunction, arg_count: int, token: Token) -> bool:
        expected_arguments = function.input_sockets()
        if len(expected_arguments) < arg_count:
            self.error_at(token,
                          f'Expected at most {len(expected_arguments)} argument(s), but got {arg_count} arguments')
            return False
        assert len(self.stack) >= arg_count, 'Bug in type checker'
        arg_types = self.stack[-arg_count:]
        self.stack = self.stack[:-arg_count]
        if not self.test_conversion_of_args(arg_types, expected_arguments):
            return False
        if arg_count < len(expected_arguments):
            # Fill up with default values
            for _ in range(len(expected_arguments)-arg_count):
                # No need to type check these.
                self.add_operation(OpType.PUSH_VALUE, None)
        return True

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
        if not self.type_check_arguments(function, arg_count, instruction.token):
            return False
        self.add_operation(OpType.CALL_FUNCTION, function)
        outputs = function.output_sockets()
        assert len(outputs) == 1, 'Not implemented'
        self.stack.append(TypeCheckValue(instruction.token, Value(
            outputs[0].sock_type, NodeSocket), True, 0))
        return True

    def type_check_assignment(self, assignment_instruction: Instruction):
        raise NotImplementedError

    def type_check(self, program: list[Instruction]) -> bool:
        """
        Go through all the instructions and make sure types and argument counts match up.
        This will modify the program in some cases to ensure correctness.
        """
        assert InstructionType.IGNORE.value == 7, 'Exhaustive handling of instructions'
        ip = 0
        while ip < len(program):
            instruction = program[ip]
            itype = instruction.instruction
            if itype == InstructionType.VALUE:
                assert isinstance(instruction.data,
                                  Value), 'Parser Bug: non-value data.'
                self.add_operation(OpType.PUSH_VALUE, instruction.data.value)
                self.stack.append(TypeCheckValue(
                    instruction.token, instruction.data, False, len(self.checked_program) - 1))
            elif itype == InstructionType.VAR:
                raise NotImplementedError
            elif itype == InstructionType.GET_VAR:
                raise NotImplementedError
            elif itype == InstructionType.PROPERTIES:
                assert False, 'Parser Bug: Properties should have been handled already.'
            elif itype == InstructionType.FUNCTION:
                if not self.type_check_function(instruction):
                    return False
            elif itype == InstructionType.END_OF_STATEMENT:
                self.stack = []
                self.add_operation(OpType.END_OF_STATEMENT, None)
            elif itype == InstructionType.ASSIGNMENT:
                raise NotImplementedError
            elif itype == InstructionType.IGNORE:
                raise NotImplementedError
            ip += 1
        return True
        # raise NotImplementedError


class Compiler():
    def __init__(self) -> None:
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []
        self.checked_program: list[Operation] = []

    def compile(self, source: str, macros: MacroType) -> bool:
        self.instructions = []
        parser = Parser(source, macros)
        # macros = parser.macro_storage
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
        parser = Parser(source, {})
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
