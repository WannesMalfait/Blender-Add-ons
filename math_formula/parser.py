import math
from bpy.types import NodeSocket
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
    STRUCT = auto()
    VAR = auto()
    GET_VARIABLE = auto()
    PROPERTIES = auto()
    FUNCTION = auto()
    NODEGROUP = auto()
    DEF_FUNCTION = auto()
    DEF_NODEGROUP = auto()
    END_OF_STATEMENT = auto()
    ASSIGNMENT = auto()
    IGNORE = auto()


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
    def __init__(self, instruction: InstructionType, data) -> None:
        """ Create an instruction of the given type. `data` should
        contain the information needed to execute the type of instruction,
        not the arguments. """
        self.instruction = instruction
        self.data = data

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
        self.type_check_stack: list[tuple[Value, Instruction]] = []

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
            InstructionType.END_OF_STATEMENT, None))

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
        name = self.previous.lexeme
        var_type = self.parse_type()

        return Instruction(InstructionType.VAR, Variable(name, Value(var_type, None)))

    def parse_variable(self, message: str) -> Instruction:
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
                    Instruction(InstructionType.IGNORE, None))
                num_vars += 1
            else:
                self.error_at_current(message)
            if not self.match(TokenType.COMMA):
                break
        if num_vars == 0:
            self.error(message)
        return Instruction(InstructionType.ASSIGNMENT, num_vars)

    def add_value_instruction(self, value: Value) -> None:
        self.instructions.append(Instruction(InstructionType.VALUE, value))
        self.type_check_stack.append((value, self.instructions[-1]))

    def type_check_assignment(self, assignment_instruction: Instruction):
        raise NotImplementedError

    def variable_declaration(self) -> None:
        assignment_instruction = self.parse_variable(
            'Expect variable name or "_" after "let".')
        if self.match(TokenType.EQUAL):
            self.expression()
        else:
            # Something like 'let x: float;' gets de-sugared to 'let x = 0.0;'
            self.add_value_instruction(Value(DataType.DEFAULT, None))
        self.instructions.append(assignment_instruction)
        self.type_check_assignment(assignment_instruction)
        # Get optional semicolon at end of expression
        self.match(TokenType.SEMICOLON)
        self.instructions.append(Instruction(
            InstructionType.END_OF_STATEMENT, None))

    def declaration(self) -> None:
        if self.match(TokenType.LET):
            self.variable_declaration()
        else:
            self.statement()

        if self.panic_mode:
            self.synchronize()

    def test_conversion_of_args(self, arg_types: list[tuple[Value, Instruction]], expected_arguments: list[Socket]) -> None:
        for i, arg_type in enumerate(arg_types):
            given = arg_type[0].data_type
            expected = expected_arguments[i].sock_type
            if given == DataType.DEFAULT or expected == given:
                continue
            elif given.can_convert(expected):
                if arg_type[0].value == NodeSocket:
                    continue
                else:
                    instruction = arg_type[1]
                    value: Value = instruction.data
                    value.convert(expected)
            else:
                self.error(
                    f'Invalid type, can\'t convert from {data_type_to_string[given]} to {data_type_to_string[expected]}.')

    def type_check_arguments(self, function: NodeFunction, arg_count: int) -> None:
        expected_arguments = function.input_sockets()
        if len(expected_arguments) < arg_count:
            self.error(
                f'Expected at most {len(expected_arguments)} argument(s), but got {arg_count} arguments')
        else:
            arg_types = self.type_check_stack[-arg_count:]
            self.type_check_stack = self.type_check_stack[:-arg_count]
            self.test_conversion_of_args(arg_types, expected_arguments)
            if arg_count < len(expected_arguments):
                for _ in range(len(expected_arguments)-arg_count):
                    # No need to type check these.
                    self.instructions.append(Instruction(
                        InstructionType.VALUE, Value(DataType.DEFAULT, None)))

    def type_check_outputs(self, function: NodeFunction) -> None:
        self.instructions.append(Instruction(
            InstructionType.FUNCTION, function))
        outputs = function.output_sockets()
        if len(outputs) == 1:
            self.type_check_stack.append(
                (Value(outputs[0].sock_type, NodeSocket), self.instructions[-1]))
        else:
            raise NotImplementedError
            # for output in outputs:
            #     self.type_check_stack.append(Value(output.sock_type, NodeSocket))

    def type_check_multiply(self) -> None:
        arg2 = self.type_check_stack.pop()
        arg1 = self.type_check_stack.pop()
        function = None
        if arg1[0].data_type == DataType.VEC3:
            if arg2[0].data_type == DataType.VEC3:
                function = function_nodes.VectorMath(['MULTIPLY'])
            else:
                function = function_nodes.VectorMath(['SCALE'])
        elif arg2[0].data_type == DataType.VEC3:
            function = function_nodes.VectorMath(['SCALE'])
            # Vector should be first input!
            instruction1 = arg1[1]
            arg2, arg1 = arg1, arg2
            self.instructions.remove(instruction1)
            self.instructions.append(instruction1)
        else:
            function = function_nodes.Math(['MULTIPLY'])
        self.test_conversion_of_args((arg1, arg2), function.input_sockets())
        self.type_check_outputs(function)

    def type_check_operator(self, operator_name: str) -> None:
        arg2 = self.type_check_stack.pop()
        arg1 = self.type_check_stack.pop()
        function = None
        if arg1[0].data_type == DataType.VEC3 or arg2[0].data_type == DataType.VEC3:
            function = function_nodes.VectorMath([operator_name])
        else:
            function = function_nodes.Math([operator_name])
        self.test_conversion_of_args((arg1, arg2), function.input_sockets())
        self.type_check_outputs(function)

    def argument_list(self, function: NodeFunction, closing_token: Token) -> None:
        arg_count = 0
        if not self.check(TokenType.RIGHT_PAREN):
            self.expression()
            arg_count += 1
            while self.match(TokenType.COMMA):
                self.expression()
                arg_count += 1
        self.consume(closing_token.token_type,
                     f'Expect "{closing_token.lexeme}" after arguments.')
        self.type_check_arguments(function, arg_count)
        self.type_check_outputs(function)

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
    name = self.previous.lexeme
    if name in self.macro_storage:
        rhs = self.macro_storage[name]
        for token in rhs:
            token.expanded_from = self.previous
        self.add_tokens(rhs)
        self.expression()
    else:
        self.instructions.append(Instruction(
            InstructionType.GET_VARIABLE, name)
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
    operator_type = self.previous.token_type
    # Compile the operand
    self.parse_precedence(Precedence.UNARY)

    if operator_type == TokenType.MINUS:
        # unary minus (-x) in shader nodes is x * -1
        # postfix: x -1 *
        self.add_value_instruction(Value(DataType.INT, -1))
        self.type_check_multiply()
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
    function = function_nodes.CombineXYZ([])
    self.argument_list(function, Token('}', TokenType.RIGHT_BRACE))
    # Check if the arguments are just regular values. In that case
    # we don't need a combine XYZ node.
    args = self.instructions[-4:-1]
    no_expressions = True
    for arg in args:
        if not isinstance(arg.data, Value):
            no_expressions = False
            break
    if no_expressions:
        self.instructions = self.instructions[:-4]
        # Get rid of the result of CombineXYZ
        self.type_check_stack.pop()
        self.add_value_instruction(
            Value(DataType.VEC3, make_vec3_from_values(args)))


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
    if prev_instruction.instruction != InstructionType.GET_VARIABLE:
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
    self.argument_list(function, Token(')', TokenType.RIGHT_PAREN))


def properties(self: Parser, can_assign: bool) -> None:
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
        # These have been handled
        self.type_check_stack.pop()
        if not self.match(TokenType.COMMA):
            break
    self.consume(TokenType.RIGHT_SQUARE_BRACKET, 'Expect closing "]".')
    self.instructions.append(Instruction(
        InstructionType.PROPERTIES, num_props))


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
    operator_type = self.previous.token_type
    rule = self.get_rule(operator_type)
    self.parse_precedence(Precedence(rule.precedence.value + 1))

    # math: + - / * % > < **

    if operator_type == TokenType.PLUS:
        self.type_check_operator('ADD')
    elif operator_type == TokenType.MINUS:
        self.type_check_operator('SUBTRACT')
    elif operator_type == TokenType.SLASH:
        self.type_check_operator('DIVIDE')
    elif operator_type == TokenType.STAR:
        self.type_check_multiply()
        # self.instructions.append(Instruction(
        #     InstructionType.FUNCTION, function_nodes.Math(['MULTIPLY'])))
    elif operator_type == TokenType.PERCENT:
        self.type_check_operator('MODULO')
    elif operator_type == TokenType.GREATER:
        function = function_nodes.Math(['GREATER_THAN'])
        self.type_check_arguments(function, 2)
    elif operator_type == TokenType.LESS:
        function = function_nodes.Math(['LESS_THAN'])
        self.type_check_arguments(function, 2)
    elif operator_type in (TokenType.STAR_STAR, TokenType.HAT):
        function = function_nodes.Math(['POWER'])
        self.type_check_arguments(function, 2)
    else:
        assert False, "Unreachable code"


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


class Compiler():
    def __init__(self) -> None:
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []

    def compile(self, source: str, macros: MacroType) -> bool:
        self.instructions = []
        parser = Parser(source, macros)
        macros = parser.macro_storage
        parser.advance()
        while not parser.match(TokenType.EOL):
            parser.declaration()
        parser.consume(TokenType.EOL, 'Expect end of expression.')
        self.instructions = parser.instructions
        self.errors = parser.errors
        return not parser.had_error

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
