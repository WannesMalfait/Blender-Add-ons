import math
from typing import Tuple
from scanner import *
from enum import IntEnum, auto
from bpy.types import NodeSocket


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


class DataType(IntEnum):
    UNKNOWN = 0
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    RGBA = auto()
    VEC3 = auto()
    GEOMETRY = auto()
    STRING = auto()


string_to_data_type = {
    '_': DataType.UNKNOWN,
    'bool': DataType.BOOL,
    'int': DataType.INT,
    'float': DataType.FLOAT,
    'rgba': DataType.RGBA,
    'vec3': DataType.VEC3,
    'geometry': DataType.GEOMETRY,
    'string': DataType.STRING,
}

data_type_to_string = {value: key for key,
                       value in string_to_data_type.items()}


class InstructionType(IntEnum):
    VALUE = 0
    STRUCT = auto()
    VAR = auto()
    GET_VARIABLE = auto()
    DEFAULT = auto()
    DEFINE = auto()
    MAKE_VECTOR = auto()
    SEPARATE = auto()
    MATH_FUNC = auto()
    VECTOR_MATH_FUNC = auto()
    OTHER_FUNC = auto()
    END_OF_STATEMENT = auto()
    ASSIGNMENT = auto()
    IGNORE = auto()


ValueType = Union[None, bool, str, int, float,
                  tuple[float], NodeSocket]


class Value():
    def __init__(self,
                 data_type: DataType,
                 value: ValueType) -> None:
        self.data_type = data_type
        self.value = value

    def __str__(self) -> str:
        return f"{self.value} ({data_type_to_string[self.data_type]})"

    def __repr__(self) -> str:
        return self.__str__()


Struct = Tuple[Value]


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

    def __init__(self, source: str) -> None:
        self.scanner = Scanner(source)
        self.current: Token = None
        self.previous: Token = None
        self.had_error: bool = False
        self.panic_mode: bool = False
        self.instructions: list[Instruction] = []
        self.errors: list[Error] = []

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
        # TODO: Better handling of errors
        self.errors.append(Error(token, f'{error} {message}'))
        self.had_error = True

    def consume(self, token_type: TokenType, message: str) -> None:
        if self.check(token_type):
            self.advance()
            return

        self.error_at_current(message)

    def advance(self) -> None:
        self.previous = self.current

        # Get tokens till not an error
        while True:
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

    def get_vector_for_assignment(self) -> Instruction:
        var1 = var2 = var3 = ''
        if self.match(TokenType.ATTRIBUTE):
            var1 = self.previous.lexeme
        self.consume(TokenType.COMMA,
                     'Expect "," in between variable names')
        if self.match(TokenType.ATTRIBUTE):
            var2 = self.previous.lexeme
        self.consume(TokenType.COMMA,
                     'Expect "," in between variable names')
        if self.match(TokenType.ATTRIBUTE):
            var3 = self.previous.lexeme
        self.consume(TokenType.RIGHT_BRACE,
                     'Expect "}" at end of variable names')
        return Instruction(InstructionType.VECTOR_VAR, (var1, var2, var3))

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
        var_type = DataType.UNKNOWN

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

    def variable_declaration(self) -> None:
        assignment_instruction = self.parse_variable(
            'Expect variable name or "_" after "let".')
        if self.match(TokenType.EQUAL):
            self.expression()
        else:
            # Something like 'let x: float;' gets de-sugared to 'let x = 0.0;'
            self.instructions.append(
                Instruction(InstructionType.DEFAULT, None))
        self.instructions.append(assignment_instruction)
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

    def argument_list(self, closing_token: Token) -> None:
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
    self.instructions.append(Instruction(
        InstructionType.VALUE, Value(DataType.INT, value)))


def make_float(self: Parser, can_assign: bool) -> None:
    value = float(self.previous.lexeme)
    self.instructions.append(Instruction(
        InstructionType.VALUE, Value(DataType.FLOAT, value)))


def python(self: Parser, can_assign: bool) -> None:
    expression = self.previous.lexeme[1:]
    value = 0
    try:
        value = eval(expression, vars(math))
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as err:
        self.error(f'Invalid python syntax: {err}.')
    self.instructions.append(Instruction(InstructionType.NUMBER, value))


def identifier(self: Parser, can_assign: bool) -> None:
    self.instructions.append(Instruction(
        InstructionType.GET_VARIABLE, self.previous.lexeme)
    )


def string(self: Parser, can_assign: bool) -> None:
    self.instructions.append(Instruction(
        InstructionType.VALUE, Value(DataType.STRING, self.previous.lexeme)
    ))


def boolean(self: Parser, can_assign: bool) -> None:
    self.instructions.append(Instruction(
        InstructionType.VALUE, Value(DataType.BOOL, bool(self.previous.lexeme))
    ))


def attribute(self: Parser, can_assign: bool) -> None:
    value = self.previous.lexeme
    if can_assign and self.match(TokenType.EQUAL):
        self.instructions.append(Instruction(InstructionType.VAR, value))
        self.expression()
        self.instructions.append(Instruction(
            InstructionType.DEFINE, InstructionType.VAR))
    else:
        self.instructions.append(Instruction(InstructionType.ATTRIBUTE, value))


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
        self.instructions.append(Instruction(
            InstructionType.VALUE, Value(DataType.INT, -1)))
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('MULTIPLY', 2)))
    else:
        # Shouldn't happen
        assert False, "Unreachable code"
        return


def make_vector(self: Parser, can_assign: bool) -> None:
    arg_count = self.argument_list(Token('}', TokenType.RIGHT_BRACE))
    if arg_count != 3:
        self.error(
            f'Expect 3 arguments to vector, got {arg_count} argument(s)')
    self.instructions.append(Instruction(InstructionType.MAKE_VECTOR, None))


def call(self: Parser, can_assign: bool) -> None:
    func = self.previous
    self.consume(TokenType.LEFT_PAREN, 'Expect "(" after function name.')
    arg_count = self.argument_list(Token(')', TokenType.RIGHT_PAREN))
    if func.token_type == TokenType.MATH_FUNC:
        instruction_type = InstructionType.MATH_FUNC
        name, expected_number_of_args = math_operations[func.lexeme]
    elif func.token_type == TokenType.VECTOR_MATH_FUNC:
        instruction_type = InstructionType.VECTOR_MATH_FUNC
        name, expected_number_of_args = vector_math_operations[func.lexeme]
    else:
        instruction_type = InstructionType.OTHER_FUNC
        name, prop, expected_number_of_args = other_functions[func.lexeme]
    if expected_number_of_args != arg_count:
        # If it is a binary function, we can deal with more arguments,
        # i.e. add(4,5,6) = add(add(4,5),6). If there are missing arguments
        # we detect those. Otherwise we give an error.
        if arg_count > expected_number_of_args and expected_number_of_args == 2:
            for _ in range(arg_count-expected_number_of_args):
                self.instructions.append(Instruction(
                    instruction_type, (name, expected_number_of_args)))
        elif arg_count < expected_number_of_args:
            for _ in range(expected_number_of_args-arg_count):
                self.instructions.append(
                    Instruction(InstructionType.MISSING_ARG, None))
        else:
            self.error_at(func,
                          f'{name} expects {expected_number_of_args} argument(s), got {arg_count} argument(s)')
    if func.token_type == TokenType.OTHER_FUNC:
        self.instructions.append(Instruction(
            instruction_type, (name, prop, expected_number_of_args)))
    else:
        self.instructions.append(Instruction(
            instruction_type, (name, expected_number_of_args)))


def separate(self: Parser, can_assign: bool) -> None:
    self.consume(TokenType.ATTRIBUTE, 'Expect "xyz" after ".".')
    self.instructions.append(Instruction(
        InstructionType.VAR, self.previous.lexeme))
    self.instructions.append(Instruction(InstructionType.SEPARATE, None))


def binary(self: Parser, can_assign: bool) -> None:
    operator_type = self.previous.token_type
    rule = self.get_rule(operator_type)
    self.parse_precedence(Precedence(rule.precedence.value + 1))

    # math: + - / * % > < **
    if operator_type == TokenType.PLUS:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('ADD', 2)))
    elif operator_type == TokenType.MINUS:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('SUBTRACT', 2)))
    elif operator_type == TokenType.SLASH:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('DIVIDE', 2)))
    elif operator_type == TokenType.STAR:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('MULTIPLY', 2)))
    elif operator_type == TokenType.PERCENT:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('MODULO', 2)))
    elif operator_type == TokenType.GREATER:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('GREATER_THAN', 2)))
    elif operator_type == TokenType.LESS:
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('LESS_THAN', 2)))
    elif operator_type in (TokenType.STAR_STAR, TokenType.HAT):
        self.instructions.append(Instruction(
            InstructionType.MATH_FUNC, ('POWER', 2)))
    # Vector math: v+ v- v/ v* v%
    elif operator_type == TokenType.VECTOR_PLUS:
        self.instructions.append(Instruction(
            InstructionType.VECTOR_MATH_FUNC, ('ADD', 2)))
    elif operator_type == TokenType.VECTOR_MINUS:
        self.instructions.append(Instruction(
            InstructionType.VECTOR_MATH_FUNC, ('SUBTRACT', 2)))
    elif operator_type == TokenType.VECTOR_SLASH:
        self.instructions.append(Instruction(
            InstructionType.VECTOR_MATH_FUNC, ('DIVIDE', 2)))
    elif operator_type == TokenType.VECTOR_STAR:
        self.instructions.append(Instruction(
            InstructionType.VECTOR_MATH_FUNC, ('MULTIPLY', 2)))
    elif operator_type == TokenType.VECTOR_PERCENT:
        self.instructions.append(Instruction(
            InstructionType.VECTOR_MATH_FUNC, ('MODULO', 2)))
    else:
        # Shouldn't happen
        return


rules: list[ParseRule] = [
    ParseRule(grouping, call, Precedence.NONE),  # LEFT_PAREN
    ParseRule(None, None, Precedence.NONE),  # RIGHT_PAREN
    ParseRule(None, None, Precedence.NONE),  # LEFT_SQUARE_BRACKET
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
    ParseRule(None, None, Precedence.NONE),  # UNDERSCORE

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
    ParseRule(None, None, Precedence.NONE),  # MACRO
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

    def compile(self, source: str) -> bool:
        self.instructions = []
        parser = Parser(source)
        parser.advance()
        while not parser.match(TokenType.EOL):
            parser.declaration()
        parser.consume(TokenType.EOL, 'Expect end of expression.')
        self.instructions = parser.instructions
        self.errors = parser.errors
        return not parser.had_error

    @staticmethod
    def get_tokens(source: str) -> list[Token]:
        tokens = []
        parser = Parser(source)
        scanner = parser.scanner
        while(token := scanner.scan_token()).token_type != TokenType.EOL:
            tokens.append(token)
        return tokens


if __name__ == '__main__':
    tests = [
        'let x;let y=2+5**-.5; let z = -y*x;',
        'let a,_,c = "HEY";',
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
