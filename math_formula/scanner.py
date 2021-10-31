from enum import IntEnum, auto
from typing import Union

other_functions = {
    # alias:
    # (bl_idname without 'ShaderNode' or 'GeometryNode',
    # props: (prop_name, value)
    # num_args
    'map': ('MapRange', ('interpolation_type', 'LINEAR'), 5),
    'smoothstep': ('MapRange', ('interpolation_type', 'SMOOTHSTEP'), 5),
    'sstep': ('MapRange', ('interpolation_type', 'SMOOTHSTEP'), 5),
    'smootherstep': ('MapRange', ('interpolation_type', 'SMOOTHERSTEP'), 5),
    'ssstep': ('MapRange', ('interpolation_type', 'SMOOTHERSTEP'), 5),
    'clamp': ('Clamp', ('clamp_type', 'MINMAX'), 3),
    'minmax': ('Clamp', ('clamp_type', 'MINMAX'), 3),
    'clamprange': ('Clamp', ('clamp_type', 'RANGE'), 3),
}


# See: https://craftinginterpreters.com/scanning-on-demand.html
# For the original source of this code

class TokenType(IntEnum):
    # Single-character tokens.
    LEFT_PAREN = 0
    RIGHT_PAREN = auto()
    LEFT_SQUARE_BRACKET = auto()
    RIGHT_SQUARE_BRACKET = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    SEMICOLON = auto()
    EQUAL = auto()
    MINUS = auto()
    PLUS = auto()
    PERCENT = auto()
    SLASH = auto()
    HAT = auto()
    GREATER = auto()
    LESS = auto()
    DOLLAR = auto()
    COLON = auto()
    UNDERSCORE = auto()

    # One or two character tokens.
    STAR = auto()
    STAR_STAR = auto()
    ARROW = auto()

    # Literals.
    IDENTIFIER = auto()
    INT = auto()
    FLOAT = auto()
    PYTHON = auto()
    STRING = auto()

    # Keywords
    LET = auto()
    FUNCTION = auto()
    NODEGROUP = auto()
    MACRO = auto()
    SELF = auto()
    TRUE = auto()
    FALSE = auto()

    ERROR = auto()
    EOL = auto()


class Token():
    """
    The parser turns the input string into a list of tokens.

    Each token has:
    - a `TokenType`
    - a lexeme which is the text that this token had in the source
    - a number line which says which line of the text the token is in
    - a number col which says where in the line the token starts
    """

    def __init__(self, lexeme: str, token_type: TokenType, line: int = 0, start: int = 0) -> None:
        self.token_type = token_type
        self.start = start
        self.line = line
        self.lexeme = lexeme
        self.expanded_from = None

    def __str__(self) -> str:
        return f'[{self.lexeme}, {self.token_type.name}]'

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, o: object) -> bool:
        if type(o) != Token:
            return False
        return self.lexeme == o.lexeme and self.token_type == o.token_type


class Scanner():
    def __init__(self, source: str) -> None:
        """ Initialize the scanner with source code to scan """
        self.reset(source)

    def reset(self, source: str) -> None:
        # Place a sentinel at the end of the string
        self.source = source + '\0'
        self.start = 0
        self.line = 1
        self.current = 0

    def is_at_end(self) -> bool:
        return self.source[self.current] == '\0'

    def advance(self) -> str:
        self.current += 1
        return self.source[self.current-1]

    def peek(self) -> str:
        return self.source[self.current]

    def peek_next(self) -> str:
        if self.is_at_end():
            return '\0'
        return self.source[self.current+1]

    def match(self, expected: str) -> bool:
        if self.is_at_end():
            return False
        if self.peek() != expected:
            return False
        self.current += 1
        return True

    def skip_whitespace(self) -> None:
        while True:
            c = self.peek()
            if c == '\n':
                self.line += 1
                self.advance()
            elif c.isspace():
                self.advance()
            else:
                return

    def make_token(self, token_type: TokenType) -> Token:
        return Token(self.source[self.start: self.current], token_type,
                     line=self.line, start=self.start)

    def error_token(self, message: str) -> Token:
        return Token((self.source[self.start: self.current], message),
                     TokenType.ERROR, line=self.line, start=self.start)

    def keyword(self) -> Union[TokenType, None]:
        """ Checks if it's a keyword, otherwise it's treated as an identifier."""
        name = self.source[self.start: self.current]
        if name == 'let':
            return self.make_token(TokenType.LET)
        if name == 'fn':
            return self.make_token(TokenType.FUNCTION)
        if name == 'ng':
            return self.make_token(TokenType.NODEGROUP)
        if name == 'MACRO':
            return self.make_token(TokenType.MACRO)
        if name == 'self':
            return self.make_token(TokenType.SELF)
        if name == 'true':
            return self.make_token(TokenType.TRUE)
        if name == 'false':
            return self.make_token(TokenType.FALSE)
        return self.make_token(TokenType.IDENTIFIER)

    def identifier(self) -> Token:
        while self.peek().isalpha() or self.peek().isdecimal() or self.peek() == '_':
            self.advance()
        return self.keyword()

    def number(self) -> Token:
        while self.peek().isdecimal():
            self.advance()
        if self.match('.'):
            return self.float()
        return self.make_token(TokenType.INT)

    def float(self) -> Token:
        """ 
        Find a floating point number from the current position.
        Everything until the '.' should have been handled already.
        """
        while self.peek().isdecimal():
            self.advance()
        return self.make_token(TokenType.FLOAT)

    def python(self) -> Token:
        """ Get the string in between () """
        if self.peek() != '(':
            # It's a single value like '!pi'.
            while self.peek().isalpha() or self.peek().isdecimal() or self.peek() == '_':
                self.advance()
            return self.make_token(TokenType.PYTHON)
        self.advance()
        open_parentheses = 1
        while not self.is_at_end() and open_parentheses != 0:
            c = self.peek()
            if c == '(':
                open_parentheses += 1
            elif c == ')':
                open_parentheses -= 1
            self.advance()
        return self.make_token(TokenType.PYTHON)

    def string(self, closing: str) -> Token:
        """ Get the string in between \' or \"."""
        while not self.is_at_end():
            if self.match(closing):
                return self.make_token(TokenType.STRING)
            self.advance()
        return self.error_token('Expected string to be closed.')

    def scan_token(self) -> Token:
        self.skip_whitespace()
        self.start = self.current

        # We have reached the end of the line
        if self.is_at_end():
            return self.make_token(TokenType.EOL)

        c = self.advance()
        if not c.isascii():
            return self.error_token('Unrecognized token')

        if c == '_':
            nextc = self.peek()
            if nextc.isalpha() or nextc.isdecimal() or nextc == '_':
                return self.identifier()
            return self.make_token(TokenType.UNDERSCORE)
        elif c.isalpha():
            return self.identifier()
        elif (c.isdecimal()):
            return self.number()

        # Check for single character tokens:
        elif c == '(':
            return self.make_token(TokenType.LEFT_PAREN)
        elif c == ')':
            return self.make_token(TokenType.RIGHT_PAREN)
        elif c == '{':
            return self.make_token(TokenType.LEFT_BRACE)
        elif c == '}':
            return self.make_token(TokenType.RIGHT_BRACE)
        elif c == '[':
            return self.make_token(TokenType.LEFT_SQUARE_BRACKET)
        elif c == ']':
            return self.make_token(TokenType.RIGHT_SQUARE_BRACKET)
        elif c == ';':
            return self.make_token(TokenType.SEMICOLON)
        elif c == ',':
            return self.make_token(TokenType.COMMA)
        elif c == '+':
            return self.make_token(TokenType.PLUS)
        elif c == '/':
            return self.make_token(TokenType.SLASH)
        elif c == '^':
            return self.make_token(TokenType.HAT)
        elif c == '>':
            return self.make_token(TokenType.GREATER)
        elif c == '<':
            return self.make_token(TokenType.LESS)
        elif c == '=':
            return self.make_token(TokenType.EQUAL)
        elif c == '$':
            return self.make_token(TokenType.DOLLAR)
        elif c == ':':
            return self.make_token(TokenType.COLON)
        # Check for two-character tokens
        elif c == '*':
            return self.make_token(
                TokenType.STAR_STAR if self.match('*') else TokenType.STAR)
        elif c == '.':
            # Check for floating point numbers like '.314'
            if self.peek().isdecimal():
                return self.float()
            return self.make_token(TokenType.DOT)
        elif c == '-':
            # Check for negative numbers
            if self.match('.'):
                return self.float()
            if self.match('>'):
                return self.make_token(TokenType.ARROW)
            elif self.peek().isdecimal():
                return self.number()
            return self.make_token(TokenType.MINUS)
        elif c == '#':
            return self.python()
        elif c == "'" or c == '"':
            return self.string(closing=c)
        return self.error_token('Unrecognized token')


if __name__ == '__main__':
    def scanner_test(source: str, expected: list[Token]) -> None:
        print('Testing:', source)
        scanner = Scanner(source)
        results = []
        while(token := scanner.scan_token()).token_type != TokenType.EOL:
            results.append(token)
        if len(results) != len(expected):
            print('TEST FAILED: lengths do not match')
            print('Parse results:', results)
            print('Expected:', expected)
            return
        for i, token in enumerate(expected):
            if results[i] == token:
                continue
            print('TEST FAILED: token mismatch')
            print('Parsed token:', results[i])
            print('Expected:', token)
            return
        print('Test passed!')

    scanner_tests = [
        ('4*.5-v **2.17', [
            Token('4', TokenType.INT),
            Token('*', TokenType.STAR),
            Token('.5', TokenType.FLOAT),
            Token('-', TokenType.MINUS),
            Token('v', TokenType.IDENTIFIER),
            Token('**', TokenType.STAR_STAR),
            Token('2.17', TokenType.FLOAT),
        ]),
        ('add(_,5)', [
            Token('add', TokenType.IDENTIFIER),
            Token('(', TokenType.LEFT_PAREN),
            Token('_', TokenType.UNDERSCORE),
            Token(',', TokenType.COMMA),
            Token('5', TokenType.INT),
            Token(')', TokenType.RIGHT_PAREN),
        ]),
        ('let z = sin(x*#(sqrt(pi)))', [
            Token('let', TokenType.LET),
            Token('z', TokenType.IDENTIFIER),
            Token('=', TokenType.EQUAL),
            Token('sin', TokenType.IDENTIFIER),
            Token('(', TokenType.LEFT_PAREN),
            Token('x', TokenType.IDENTIFIER),
            Token('*', TokenType.STAR),
            Token('#(sqrt(pi))', TokenType.PYTHON),
            Token(')', TokenType.RIGHT_PAREN),
        ]),
        ('let string = "TESKJH49220P2%£¨2";', [
            Token('let', TokenType.LET),
            Token('string', TokenType.IDENTIFIER),
            Token('=', TokenType.EQUAL),
            Token('"TESKJH49220P2%£¨2"', TokenType.STRING),
            Token(';', TokenType.SEMICOLON),
        ]),
        ('fn test(x: float) -> y: float {self.y = sin(x) * x;}', [
            Token('fn', TokenType.FUNCTION),
            Token('test', TokenType.IDENTIFIER),
            Token('(', TokenType.LEFT_PAREN),
            Token('x', TokenType.IDENTIFIER),
            Token(':', TokenType.COLON),
            Token('float', TokenType.IDENTIFIER),
            Token(')', TokenType.RIGHT_PAREN),
            Token('->', TokenType.ARROW),
            Token('y', TokenType.IDENTIFIER),
            Token(':', TokenType.COLON),
            Token('float', TokenType.IDENTIFIER),
            Token('{', TokenType.LEFT_BRACE),
            Token('self', TokenType. SELF),
            Token('.', TokenType.DOT),
            Token('y', TokenType.IDENTIFIER),
            Token('=', TokenType.EQUAL),
            Token('sin', TokenType.IDENTIFIER),
            Token('(', TokenType.LEFT_PAREN),
            Token('x', TokenType.IDENTIFIER),
            Token(')', TokenType.RIGHT_PAREN),
            Token('*', TokenType.STAR),
            Token('x', TokenType.IDENTIFIER),
            Token(';', TokenType.SEMICOLON),
            Token('}', TokenType.RIGHT_BRACE)
        ])
    ]
    for source, expected in scanner_tests:
        scanner_test(source, expected)
