from enum import IntEnum, auto
from typing import Union


math_operations = {
    # keyword : (function name, number of arguments)
    'add': ('ADD', 2),
    'sub': ('SUBTRACT', 2),
    'mul': ('MULTIPLY', 2),
    'mult': ('MULTIPLY', 2),
    'div': ('DIVIDE', 2),
    'mult_add': ('MULTIPLY_ADD', 3),
    'sin': ('SINE', 1),
    'sine': ('SINE', 1),
    'cos': ('COSINE', 1),
    'cosine': ('COSINE', 1),
    'tan': ('TANGENT', 1),
    'tangent': ('TANGENT', 1),
    'asin': ('ARCSINE', 1),
    'arcsin': ('ARCSINE', 1),
    'arcsine': ('ARCSINE', 1),
    'acos': ('ARCCOSINE', 1),
    'arccos': ('ARCCOSINE', 1),
    'arccosine': ('ARCCOSINE', 1),
    'atan': ('ARCTANGENT', 1),
    'arctan': ('ARCTANGENT', 1),
    'arctangent': ('ARCTANGENT', 1),
    'atan2': ('ARCTAN2', 2),
    'arctan2': ('ARCTAN2', 2),
    'sinh': ('SINH', 1),
    'cosh': ('COSH', 1),
    'tanh': ('TANH', 1),
    'pow': ('POWER', 2),
    'power': ('POWER', 2),
    'log': ('LOGARITHM', 2),
    'logarithm': ('LOGARITHM', 2),
    'sqrt': ('SQRT', 1),
    'inv_sqrt': ('INVERSE_SQRT', 1),
    'exp': ('EXPONENT', 1),
    'min': ('MINIMUM', 2),
    'minimum': ('MINIMUM', 2),
    'max': ('MAXIMUM', 2),
    'maximum': ('MAXIMUM', 2),
    'less_than': ('LESS_THAN', 2),
    'greater_than': ('GREATER_THAN', 2),
    'sgn': ('SIGN', 1),
    'sign': ('SIGN', 1),
    'compare': ('COMPARE', 3),
    'smin': ('SMOOTH_MIN', 3),
    'smooth_min': ('SMOOTH_MIN', 3),
    'smooth_minimum': ('SMOOTH_MIN', 3),
    'smax': ('SMOOTH_MAX', 3),
    'smooth_max': ('SMOOTH_MAX', 3),
    'smooth_maximum': ('SMOOTH_MAX', 3),
    'fract': ('FRACT', 1),
    'mod': ('MODULO', 2),
    'snap': ('SNAP', 2),
    'wrap': ('WRAP', 3),
    'pingpong': ('PINGPONG', 2),
    'ping_pong': ('PINGPONG', 2),
    'abs': ('ABSOLUTE', 1),
    'absolute': ('ABSOLUTE', 1),
    'round': ('ROUND', 1),
    'floor': ('FLOOR', 1),
    'ceil': ('CEIL', 1),
    'trunc': ('TRUNCATE', 1),
    'truncate': ('TRUNCATE', 1),
    'rad': ('RADIANS', 1),
    'to_rad': ('RADIANS', 1),
    'to_radians': ('RADIANS', 1),
    'radians': ('RADIANS', 1),
    'deg': ('DEGREES', 1),
    'to_deg': ('DEGREES', 1),
    'to_degrees': ('DEGREES', 1),
    'degrees': ('DEGREES', 1),
}

vector_math_operations = {
    'vadd': ('ADD', 2),
    'vsub': ('SUBTRACT', 2),
    'vmult': ('MULTIPLY', 2),
    'vdiv': ('DIVIDE', 2),
    'vcross': ('CROSS_PRODUCT', 2),
    'cross': ('CROSS_PRODUCT', 2),
    'cross_product': ('CROSS_PRODUCT', 2),
    'vproject': ('PROJECT', 2),
    'project': ('PROJECT', 2),
    'vreflect': ('REFLECT', 2),
    'reflect': ('REFLECT', 2),
    'refract': ('REFRACT', 3),
    'vrefract': ('REFRACT', 3),
    'vfaceforward': ('FACEFORWARD', 3),
    'faceforward': ('FACEFORWARD', 3),
    'vsnap': ('SNAP', 2),
    'vmod': ('MODULO', 2),
    'vmin': ('MINIMUM', 2),
    'vminimum': ('MINIMUM', 2),
    'vmax': ('MAXIMUM', 2),
    'vmaximum': ('MAXIMUM', 2),
    'vdot': ('DOT_PRODUCT', 2),
    'dot': ('DOT_PRODUCT', 2),
    'dot_product': ('DOT_PRODUCT', 2),
    'vdist': ('DISTANCE', 2),
    'dist': ('DISTANCE', 2),
    'distance': ('DISTANCE', 2),
    'vlength': ('LENGTH', 1),
    'length': ('LENGTH', 1),
    'vscale': ('SCALE', 2),
    'scale': ('SCALE', 2),
    'vnormalize': ('NORMALIZE', 1),
    'normalize': ('NORMALIZE', 1),
    'vfloor': ('FLOOR', 1),
    'vceil': ('CEIL', 1),
    'vfract': ('FRACTION', 1),
    'vabs': ('ABSOLUTE', 1),
    'vabsolute': ('ABSOLUTE', 1),
    'vsin': ('SINE', 1),
    'vsine': ('SINE', 1),
    'vcos': ('COSINE', 1),
    'vcosine': ('COSINE', 1),
    'vtan': ('TANGENT', 1),
    'vtangent': ('TANGENT', 1),
    'vwrap': ('WRAP', 3),
}

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
    - a number start which says where in the source the token starts
    """

    def __init__(self, lexeme: str, token_type: TokenType, start: int = 0) -> None:
        self.token_type = token_type
        self.start = start
        self.lexeme = lexeme

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
            if c.isspace():
                self.advance()
            else:
                return

    def make_token(self, token_type: TokenType) -> Token:
        return Token(self.source[self.start: self.current], token_type, start=self.start)

    def error_token(self, message: str) -> Token:
        return Token((self.source[self.start: self.current], message), TokenType.ERROR, start=self.start)

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
