from enum import IntEnum, auto
from typing import Union


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
    COLON = auto()
    UNDERSCORE = auto()

    # One or two character tokens.
    STAR = auto()
    STAR_STAR = auto()
    ARROW = auto()
    LESS_EQUAL = auto()
    GREATER_EQUAL = auto()
    EQUAL_EQUAL = auto()

    # Literals.
    IDENTIFIER = auto()
    INT = auto()
    FLOAT = auto()
    PYTHON = auto()
    STRING = auto()
    GROUP_NAME = auto()

    # Keywords
    OUT = auto()
    FUNCTION = auto()
    NODEGROUP = auto()
    LOOP = auto()
    TRUE = auto()
    FALSE = auto()
    NOT = auto()
    OR = auto()
    AND = auto()

    ERROR = auto()
    EOL = auto()


class Token():
    """
    The parser turns the input string into a list of tokens.

    Each token has:
    - a `TokenType`
    - a `lexeme` which is the text that this token had in the source
    - a number `line` which says which line of the text the token is in
    - a number `col` which says where in the line the token starts
    - a number `start` which says where in the text the token starts
    """

    def __init__(self, lexeme: str, token_type: TokenType, line: int = 0, col: int = 0, start: int = 0) -> None:
        self.token_type = token_type
        self.start = start
        self.line = line
        self.col = col
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
        self.line = 1
        self.col = 1
        self.current = 0

    def is_at_end(self) -> bool:
        return self.source[self.current] == '\0'

    def advance(self) -> str:
        self.current += 1
        self.col += 1
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
        self.col += 1
        return True

    def skip_whitespace(self) -> None:
        while True:
            c = self.peek()
            if c == '\n':
                self.line += 1
                self.col = 0
                self.advance()
            elif c.isspace():
                self.advance()
            else:
                return

    def make_token(self, token_type: TokenType) -> Token:
        col = self.col - (self.current-self.start)
        return Token(self.source[self.start: self.current], token_type,
                     line=self.line, col=col, start=self.start)

    def error_token(self, message: str) -> Token:
        col = self.col - (self.current-self.start)
        return Token((self.source[self.start: self.current], message),
                     TokenType.ERROR, line=self.line, col=col, start=self.start)

    def keyword(self) -> Union[TokenType, None]:
        """ Checks if it's a keyword, otherwise it's treated as an identifier."""
        name = self.source[self.start: self.current]
        if name == 'out':
            return self.make_token(TokenType.OUT)
        if name == 'fn':
            return self.make_token(TokenType.FUNCTION)
        if name == 'ng':
            return self.make_token(TokenType.NODEGROUP)
        if name == 'loop':
            return self.make_token(TokenType.LOOP)
        if name == 'true':
            return self.make_token(TokenType.TRUE)
        if name == 'false':
            return self.make_token(TokenType.FALSE)
        if name == 'not':
            return self.make_token(TokenType.NOT)
        if name == 'or':
            return self.make_token(TokenType.OR)
        if name == 'and':
            return self.make_token(TokenType.AND)
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
            # It's a single value like '#pi'.
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

    def comment(self) -> Token:
        while not self.is_at_end() and not self.match('\n'):
            self.advance()
        # We just return the next token
        return self.scan_token()

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
            next = self.peek()
            if next.isalpha() or next.isdecimal() or next == '_':
                return self.identifier()
            return self.make_token(TokenType.UNDERSCORE)
        if c == 'n':
            # Check for node group names: n"A name for the group"
            next = self.peek()
            if next == "'" or next == '"':
                self.advance()
                token = self.string(next)
                token.token_type = TokenType.GROUP_NAME
                return token
        if c.isalpha():
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
        elif c == '^':
            return self.make_token(TokenType.HAT)
        elif c == ':':
            return self.make_token(TokenType.COLON)
        # Check for two-character tokens
        elif c == '/':
            if self.match('/'):
                return self.comment()
            return self.make_token(TokenType.SLASH)
        elif c == '>':
            return self.make_token(TokenType.GREATER_EQUAL if self.match('=') else TokenType.GREATER)
        elif c == '<':
            return self.make_token(TokenType.LESS_EQUAL if self.match('=') else TokenType.LESS)
        elif c == '=':
            return self.make_token(TokenType.EQUAL_EQUAL if self.match('=') else TokenType.EQUAL)
        elif c == '*':
            return self.make_token(
                TokenType.STAR_STAR if self.match('*') else TokenType.STAR)
        elif c == '.':
            # Check for floating point numbers like '.314'
            if self.peek().isdecimal():
                return self.float()
            return self.make_token(TokenType.DOT)
        elif c == '-':
            if self.match('>'):
                return self.make_token(TokenType.ARROW)
            return self.make_token(TokenType.MINUS)
        elif c == '#':
            return self.python()
        elif c == "'" or c == '"':
            return self.string(closing=c)
        return self.error_token('Unrecognized token')


if __name__ == '__main__':
    import os
    add_on_dir = os.path.dirname(
        os.path.realpath(__file__))

    test_directory = os.path.join(add_on_dir, 'tests')
    filenames = os.listdir(test_directory)
    for filename in filenames:
        print(f'\nTesting: "{filename}"')
        with open(os.path.join(test_directory, filename), 'r') as f:
            scanner = Scanner(f.read())
            tokens = []
            while(token := scanner.scan_token()).token_type != TokenType.EOL:
                tokens.append(token)
            print(tokens)
