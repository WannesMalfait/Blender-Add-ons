import math
from typing import Union
from scanner import TokenType, Token, Scanner
from nodes.base import string_to_data_type, DataType
from enum import IntEnum, auto
import ast_defs


class Precedence(IntEnum):
    NONE = 0
    ASSIGNMENT = auto()  # =
    OR = auto()  # or
    AND = auto()  # and
    NOT = auto()    # not
    COMPARISON = auto()  # < > <= >= ==
    TERM = auto()       # + -
    FACTOR = auto()     # * / %
    UNARY = auto()      # -
    EXPONENT = auto()   # ^ **
    ATTRIBUTE = auto()  # .
    CALL = auto()       # () {}
    PRIMARY = auto()


class Error():
    def __init__(self, token: Token, message: str) -> None:
        self.token = token
        self.message = message

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return f'{self.message}'


class ParseRule():
    def __init__(self, prefix, infix, precedence: Precedence) -> None:
        self.prefix = prefix
        self.infix = infix
        self.precedence = precedence


class Parser():

    def __init__(self, source: str) -> None:
        self.scanner = Scanner(source)
        self.token_buffer: list[Token] = []
        self.current: Token = None
        self.previous: Token = None
        self.had_error: bool = False
        self.panic_mode: bool = False
        self.curr_node: Union[ast_defs.expr, None] = None
        self.errors: list[Error] = []

    def parse(self) -> ast_defs.Module:
        module = ast_defs.Module(None)
        self.advance()
        while not self.match(TokenType.EOL):
            module.body.append(self.declaration())
        return module

    def error_at_current(self, message: str) -> None:
        self.error_at(self.current, message)

    def error(self, message: str) -> None:
        self.error_at(self.previous, message)

    def error_at(self, token: Token, message: str) -> None:
        if self.panic_mode:
            return
        self.panic_mode = True
        error = f'line:{token.line}:{token.col}: Error'
        if token.token_type == TokenType.EOL:
            error += ' at end:'
        elif token.token_type == TokenType.ERROR:
            pass
        else:
            error += f' at "{token.lexeme}":'
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
            token = self.current
            token_str, message = token.lexeme
            self.error_at(Token(token_str, TokenType.ERROR,
                                line=token.line,
                                col=token.col,
                                start=token.start), message)

    def get_rule(self, token_type: TokenType) -> ParseRule:
        return rules[token_type.value]

    def parse_precedence(self, precedence: Precedence) -> ast_defs.expr:
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
        assert self.curr_node is not None, "Parse precedence should always return something"
        return self.curr_node

    def check(self, token_type: TokenType) -> bool:
        return self.current.token_type == token_type

    def match(self, token_type: TokenType) -> bool:
        if not self.check(token_type):
            return False
        self.advance()
        return True

    def expression(self) -> ast_defs.expr:
        return self.parse_precedence(Precedence.ASSIGNMENT)

    def statement(self) -> ast_defs.stmt:
        node = self.expression()
        # Get optional semicolon at end of expression
        self.match(TokenType.SEMICOLON)
        self.curr_node = None
        return node

    def parse_type(self) -> DataType:
        self.consume(TokenType.COLON, 'Expected type after argument name.')
        if self.match(TokenType.IDENTIFIER):
            if self.previous.lexeme in string_to_data_type:
                return string_to_data_type[self.previous.lexeme]
            else:
                self.error(f'Invalid data type: {self.previous.lexeme}.')
        else:
            self.error('Expected a data type')
        return DataType.UNKNOWN

    def parse_arg(self) -> ast_defs.arg:
        token = self.previous
        name = token.lexeme
        var_type = self.parse_type()
        default = None
        if self.match(TokenType.EQUAL):
            default = self.expression()
        return ast_defs.arg(token, name, var_type, default)

    def out(self) -> ast_defs.Out:
        # Something like:
        # out x = 10;
        # out x,y,z = 10;
        # out x,_,z = position();
        token = self.previous
        targets = []
        message = 'Expect variable name or "_" after "out".'
        while not self.check(TokenType.EQUAL) or self.check(TokenType.SEMICOLON):
            if self.match(TokenType.IDENTIFIER):
                targets.append(ast_defs.Name(
                    self.previous, self.previous.lexeme))
            elif self.match(TokenType.UNDERSCORE):
                targets.append(None)
            else:
                self.error_at_current(message)
            if not self.match(TokenType.COMMA):
                break
        if targets == []:
            self.error_at(token, message)
        value = self.expression()
        self.match(TokenType.SEMICOLON)  # Optional semicolon
        self.curr_node = None
        return ast_defs.Out(token, targets, value)

    def declaration(self) -> ast_defs.stmt:
        node: ast_defs.stmt = None
        if self.match(TokenType.OUT):
            node = self.out()
        elif self.match(TokenType.FUNCTION):
            raise NotImplementedError()
        elif self.match(TokenType.NODEGROUP):
            raise NotImplementedError()
        elif self.match(TokenType.LOOP):
            raise NotImplementedError()
        else:
            node = self.statement()

        if self.panic_mode:
            self.synchronize()
        return node

    def argument_list(self, closing_token: Token) -> int:
        arg_count = 0
        if not self.check(closing_token.token_type):
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
            if self.current.token_type in (TokenType.OUT, TokenType.FUNCTION, TokenType.NODEGROUP):
                return
            self.advance()


def make_int(self: Parser, can_assign: bool) -> None:
    token = self.previous
    value = int(token.lexeme)
    self.curr_node = ast_defs.Constant(token, value, DataType.INT)


def make_float(self: Parser, can_assign: bool) -> None:
    token = self.previous
    value = float(token.lexeme)
    self.curr_node = ast_defs.Constant(token, value, DataType.FLOAT)


def python(self: Parser, can_assign: bool) -> None:
    token = self.previous
    expression = token.lexeme[1:]
    value = 0
    try:
        value = eval(expression, vars(math))
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as err:
        self.error(f'Invalid python syntax: {err}.')
    try:
        value = float(value)
    except:
        self.error(f'Expected result of python expression to be a number.')
    self.curr_node = ast_defs.Constant(token, value, DataType.FLOAT)


def default(self: Parser, can_assign: bool) -> None:
    self.curr_node = ast_defs.Constant(self.previous, None, DataType.DEFAULT)


def identifier(self: Parser, can_assign: bool) -> None:
    if can_assign:
        raise NotImplementedError()
    identifier_token = self.previous
    name = identifier_token.lexeme
    self.curr_node = ast_defs.Name(identifier_token, name)


def string(self: Parser, can_assign: bool) -> None:
    token = self.previous
    self.curr_node = ast_defs.Constant(
        token, token.lexeme[1:-1], DataType.STRING)


def boolean(self: Parser, can_assign: bool) -> None:
    token = self.previous
    self.curr_node = ast_defs.Constant(
        token, token.lexeme == 'true', DataType.BOOL)


def grouping(self: Parser, can_assign: bool) -> None:
    self.curr_node = self.expression()
    self.consume(TokenType.RIGHT_PAREN, 'Expect closing ")" after expression.')


def unary(self: Parser, can_assign: bool) -> None:
    operator_token = self.previous
    operator_type = operator_token.token_type
    # Compile the operand
    operand = self.parse_precedence(Precedence.UNARY)

    unaryop = None
    if operator_type == TokenType.MINUS:
        unaryop = ast_defs.USub(operator_token)
    elif operator_type == TokenType.NOT:
        unaryop = ast_defs.Not(operator_token)
    else:
        # Shouldn't happen
        assert False, "Unreachable code"

    self.curr_node = ast_defs.UnaryOp(operator_token, unaryop, operand)


def make_vector(self: Parser, can_assign: bool) -> None:
    bracket_token = self.previous
    x = y = z = None
    if not self.match(TokenType.RIGHT_BRACE):
        x = self.expression()
        if self.match(TokenType.COMMA):
            y = self.expression()
        if self.match(TokenType.COMMA):
            z = self.expression()
        self.consume(TokenType.RIGHT_BRACE, 'Expect closing "}".')
    self.curr_node = ast_defs.Vec3(bracket_token, x, y, z)


def group_name(self: Parser, can_assign: bool) -> None:
    raise NotImplementedError()


def call(self: Parser, can_assign: bool) -> None:
    raise NotImplementedError()


def dot(self: Parser, can_assign: bool) -> None:
    token = self.previous
    self.consume(TokenType.IDENTIFIER,
                 'Expect output name or function call after ".".')
    identifier_token = self.previous
    value = self.curr_node
    self.curr_node = ast_defs.Attribute(token, value, identifier_token.lexeme)


def binary(self: Parser, can_assign: bool) -> None:
    operator_token = self.previous
    operator_type = operator_token.token_type
    left = self.curr_node
    rule = self.get_rule(operator_type)
    right = self.parse_precedence(Precedence(rule.precedence.value + 1))

    # math: + - / * % > < **
    operation = None
    if operator_type == TokenType.PLUS:
        operation = ast_defs.Add(operator_token)
    elif operator_type == TokenType.MINUS:
        operation = ast_defs.Sub(operator_token)
    elif operator_type == TokenType.SLASH:
        operation = ast_defs.Div(operator_token)
    elif operator_type == TokenType.STAR:
        operation = ast_defs.Mult(operator_token)
    elif operator_type == TokenType.PERCENT:
        operation = ast_defs.Mod(operator_token)
    elif operator_type in (TokenType.STAR_STAR, TokenType.HAT):
        operation = ast_defs.Pow(operator_token)
    elif operator_type == TokenType.GREATER:
        operation = ast_defs.Gt(operator_token)
    elif operator_type == TokenType.GREATER_EQUAL:
        operation = ast_defs.GtE(operator_token)
    elif operator_type == TokenType.LESS:
        operation = ast_defs.Lt(operator_token)
    elif operator_type == TokenType.LESS_EQUAL:
        operation = ast_defs.LtE(operator_token)
    elif operator_type == TokenType.EQUAL_EQUAL:
        operation = ast_defs.Eq(operator_token)
    elif operator_type == TokenType.BANG_EQUAL:
        operation = ast_defs.NotEq(operator_token)
    elif operator_type == TokenType.AND:
        operation = ast_defs.And(operator_token)
    elif operator_type == TokenType.OR:
        operation = ast_defs.Or(operator_token)
    else:
        assert False, "Unreachable code"
    self.curr_node = ast_defs.BinOp(operator_token, left, operation, right)


rules: list[ParseRule] = [
    ParseRule(grouping, call, Precedence.CALL),  # LEFT_PAREN
    ParseRule(None, None, Precedence.NONE),  # RIGHT_PAREN
    ParseRule(None, None, Precedence.NONE),  # LEFT_SQUARE_BRACKET
    ParseRule(None, None, Precedence.NONE),  # RIGHT_SQUARE_BRACKET
    ParseRule(make_vector, None, Precedence.NONE),  # LEFT_BRACE
    ParseRule(None, None, Precedence.NONE),  # RIGHT_BRACE
    ParseRule(None, None, Precedence.NONE),  # COMMA
    ParseRule(None, dot, Precedence.ATTRIBUTE),  # DOT
    ParseRule(None, None, Precedence.NONE),  # SEMICOLON
    ParseRule(None, None, Precedence.NONE),  # EQUAL
    ParseRule(unary, binary, Precedence.TERM),  # MINUS
    ParseRule(None, binary, Precedence.TERM),  # PLUS
    ParseRule(None, binary, Precedence.FACTOR),  # PERCENT
    ParseRule(None, binary, Precedence.FACTOR),  # SLASH
    ParseRule(None, binary, Precedence.EXPONENT),  # HAT
    ParseRule(None, binary, Precedence.COMPARISON),  # GREATER
    ParseRule(None, binary, Precedence.COMPARISON),  # LESS
    ParseRule(None, None, Precedence.NONE),  # COLON
    ParseRule(default, None, Precedence.NONE),  # UNDERSCORE

    ParseRule(None, binary, Precedence.FACTOR),  # STAR
    ParseRule(None, binary, Precedence.EXPONENT),  # STAR_STAR
    ParseRule(None, None, Precedence.NONE),  # ARROW
    ParseRule(None, binary, Precedence.COMPARISON),  # LESS_EQUAL
    ParseRule(None, binary, Precedence.COMPARISON),  # GREATER_EQUAL
    ParseRule(None, binary, Precedence.COMPARISON),  # EQUAL_EQUAL
    ParseRule(None, binary, Precedence.COMPARISON),  # BANG_EQUAL


    ParseRule(identifier, None, Precedence.NONE),  # IDENTIFIER
    ParseRule(make_int, None, Precedence.NONE),  # INT
    ParseRule(make_float, None, Precedence.NONE),  # FLOAT
    ParseRule(python, None, Precedence.NONE),  # PYTHON
    ParseRule(string, None, Precedence.NONE),  # STRING
    ParseRule(group_name, None, Precedence.NONE),  # GROUP_NAME

    ParseRule(None, None, Precedence.NONE),  # OUT
    ParseRule(None, None, Precedence.NONE),  # FUNCTION
    ParseRule(None, None, Precedence.NONE),  # NODEGROUP
    ParseRule(None, None, Precedence.NONE),  # LOOP
    ParseRule(boolean, None, Precedence.NONE),  # TRUE
    ParseRule(boolean, None, Precedence.NONE),  # FALSE
    ParseRule(unary, None, Precedence.NOT),  # NOT
    ParseRule(None, binary, Precedence.OR),  # OR
    ParseRule(None, binary, Precedence.AND),  # AND

    ParseRule(None, None, Precedence.NONE),  # ERROR
    ParseRule(None, None, Precedence.NONE),  # EOL
]
assert len(rules) == TokenType.EOL.value + 1, "Didn't handle all tokens!"


if __name__ == '__main__':
    import os
    add_on_dir = os.path.dirname(
        os.path.realpath(__file__))

    test_directory = os.path.join(add_on_dir, 'tests')
    filenames = os.listdir(test_directory)
    for filename in filenames:
        print(f'\nTesting: "{filename}"')
        with open(os.path.join(test_directory, filename), 'r') as f:
            parser = Parser(f.read())
            try:
                print(ast_defs.dump(parser.parse(), indent='.'))
            except NotImplementedError:
                print('Parsing failed')
