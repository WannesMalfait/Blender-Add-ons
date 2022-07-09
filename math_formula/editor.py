import os
import blf
import bpy
from collections import deque
from math_formula.scanner import Scanner, Token, TokenType
from math_formula.compiler import Error
from math_formula.backends.main import string_to_data_type


add_on_dir = os.path.dirname(
    os.path.realpath(__file__))

font_directory = os.path.join(add_on_dir, 'fonts')
fonts = {
    'bfont': 0,
    'regular': blf.load(os.path.join(font_directory, 'Anonymous_Pro.ttf')),
    'italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_I.ttf')),
    'bold': blf.load(os.path.join(font_directory, 'Anonymous_Pro_0.ttf')),
    'bold_italic': blf.load(os.path.join(font_directory, 'Anonymous_Pro_BI.ttf')),
}


class Editor():
    def __init__(self, pos: tuple[float, float]) -> None:
        self.pos = pos
        self.lines: list[str] = [""]
        self.line_tokens: list[list[Token]] = [[]]
        self.cursor_col: int = 0
        self.cursor_row: int = 0
        self.draw_cursor_col: int = 0
        self.scanner = Scanner("")
        self.errors: list[Error] = []
        self.suggestions: deque[str] = deque()

    # def try_auto_complete(self, tree_type: str) -> None:
    #     token_under_cursor = None
    #     prev_token = None
    #     for token in self.line_tokens[self.cursor_row]:
    #         if token.start < self.draw_cursor_col <= token.start + len(token.lexeme):
    #             token_under_cursor = token
    #             break
    #         prev_token = token
    #     if token_under_cursor is not None:
    #         if len(self.suggestions) != 0:
    #             suggestion = self.suggestions.popleft()
    #             self.replace_token(token_under_cursor, suggestion)
    #             self.suggestions.append(suggestion)
    #             return
    #         if prev_token is not None and prev_token.lexeme == '.' or token_under_cursor.lexeme == '.':
    #             token_text = token_under_cursor.lexeme
    #             text_start = token_under_cursor.start
    #             if token_under_cursor.lexeme == '.':
    #                 token_text = ''
    #                 text_start += 1
    #             parser = Parser(self.get_text()[
    #                             :text_start], file_loading.file_data.macros, tree_type)
    #             parser.advance()
    #             while not parser.match(TokenType.EOL):
    #                 parser.declaration()
    #             parser.consume(TokenType.EOL, 'Expect end of expression.')
    #             for i in range(len(parser.instructions)):
    #                 if parser.instructions[-i-1].instruction == InstructionType.GET_OUTPUT:
    #                     prev = parser.instructions[-i-2]
    #                     if prev and prev.instruction == InstructionType.FUNCTION:
    #                         assert isinstance(prev.data, tuple), 'Parser bug'
    #                         function, _ = prev.data
    #                         assert isinstance(
    #                             function, NodeFunction), 'Parser bug'
    #                         outputs = function.output_sockets()
    #                         if len(outputs) == 1:
    #                             # TODO: Make sugggestions depend on the type here.
    #                             # Not very necessary but could be nice.
    #                             pass
    #                         else:
    #                             for socket in function.output_sockets():
    #                                 if socket.name.startswith(token_text):
    #                                     self.suggestions.append(socket.name)
    #                             if len(self.suggestions) == 0:
    #                                 return
    #                             else:
    #                                 suggestion = self.suggestions.popleft()
    #                                 if token_text == '':
    #                                     self.text_after_cursor(suggestion)
    #                                 else:
    #                                     self.replace_token(
    #                                         token_under_cursor, suggestion)
    #                                 self.suggestions.append(suggestion)
    #                                 return
    #                     break
    #             if token_text == '':
    #                 # Don't try to suggest everything
    #                 return
    #         for name in file_loading.file_data.macros.keys():
    #             if name.startswith(token_under_cursor.lexeme):
    #                 self.suggestions.append(name)

    #         for name in function_nodes.functions.keys():
    #             if name.startswith(token_under_cursor.lexeme):
    #                 self.suggestions.append(name)
    #         names = None
    #         if tree_type == 'GeometryNodeTree':
    #             names = geometry_nodes.functions.keys()
    #         else:
    #             names = shader_nodes.functions.keys()
    #         for name in names:
    #             if name.startswith(token_under_cursor.lexeme):
    #                 self.suggestions.append(name)
    #         if len(self.suggestions) == 0:
    #             return
    #         suggestion = self.suggestions.popleft()
    #         self.replace_token(token_under_cursor, suggestion)
    #         self.suggestions.append(suggestion)

    def text_after_cursor(self, text: str) -> None:
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
            text + line[self.draw_cursor_col:]
        self.draw_cursor_col += len(text)
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def replace_token(self, token: Token, text: str) -> None:
        start = token.start
        end = start + len(token.lexeme)
        line = self.lines[self.cursor_row]
        first = line[:start] + text
        self.draw_cursor_col = len(first)
        self.cursor_col = self.draw_cursor_col
        self.lines[self.cursor_row] = first + line[end:]
        self.rescan_line()

    def cursor_up(self) -> None:
        self.suggestions.clear()
        if self.cursor_row == 0:
            return
        else:
            self.cursor_row -= 1
            # Make sure that we don't draw outside of the line, but
            # at the same time keep track of where the actual cursor is
            # in case we jump to a longer line next.
            self.draw_cursor_col = min(
                len(self.lines[self.cursor_row]), self.cursor_col)

    def cursor_down(self) -> None:
        self.suggestions.clear()
        if self.cursor_row == len(self.lines) - 1:
            return
        else:
            self.cursor_row += 1
            # Make sure that we don't draw outside of the line, but
            # at the same time keep track of where the actual cursor is
            # in case we jump to a longer line next.
            self.draw_cursor_col = min(
                len(self.lines[self.cursor_row]), self.cursor_col)

    def cursor_left(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == 0:
            if self.cursor_row != 0:
                self.cursor_row -= 1
                self.draw_cursor_col = len(self.lines[self.cursor_row])
        else:
            self.draw_cursor_col -= 1
        self.cursor_col = self.draw_cursor_col

    def cursor_right(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == len(self.lines[self.cursor_row]):
            if self.cursor_row != len(self.lines) - 1:
                self.cursor_row += 1
                self.draw_cursor_col = 0
        else:
            self.draw_cursor_col += 1
        self.cursor_col = self.draw_cursor_col

    def cursor_home(self) -> None:
        self.suggestions.clear()
        self.draw_cursor_col = 0
        self.cursor_col = self.draw_cursor_col

    def cursor_end(self) -> None:
        self.suggestions.clear()
        self.draw_cursor_col = len(self.lines[self.cursor_row])
        self.cursor_col = self.draw_cursor_col

    def delete_before_cursor(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col == 0:
            if self.cursor_row == 0:
                self.cursor_col = 0
                return
            # Merge this line with previous one.
            self.draw_cursor_col = len(self.lines[self.cursor_row - 1])
            self.lines[self.cursor_row-1] += self.lines[self.cursor_row]
            self.cursor_row -= 1
            self.rescan_line()
            self.cursor_col = self.draw_cursor_col
            self.lines.pop(self.cursor_row+1)
            self.line_tokens.pop(self.cursor_row+1)
            return
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.draw_cursor_col -
                                           1] + line[self.draw_cursor_col:]
        self.draw_cursor_col -= 1
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def delete_after_cursor(self) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        self.cursor_col = self.draw_cursor_col
        if self.draw_cursor_col == len(line):
            if self.cursor_row == len(self.lines) - 1:
                return
            # Merge this next line with this one.
            self.lines[self.cursor_row] += self.lines[self.cursor_row + 1]
            self.rescan_line()
            self.lines.pop(self.cursor_row+1)
            self.line_tokens.pop(self.cursor_row+1)
            return
        self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
            line[self.draw_cursor_col + 1:]
        self.rescan_line()

    def paste_after_cursor(self, text: str) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        if (index := text.find('\n')) != -1:
            self.lines[self.cursor_row] = line[:self.draw_cursor_col] + text[:index]
            self.rescan_line()
            line = line[self.draw_cursor_col:]
            text = text[index+1:]
            self.draw_cursor_col = len(self.lines[self.cursor_row])
            self.new_line()
            while True:
                if text == "":
                    break
                if (index := text.find('\n')) != -1:
                    self.lines[self.cursor_row] = text[:index]
                    self.rescan_line()
                    text = text[index+1:]
                    self.draw_cursor_col = len(self.lines[self.cursor_row])
                    self.new_line()
                else:
                    self.lines[self.cursor_row] = text + line
                    self.draw_cursor_col = 0
                    self.rescan_line()
                    break
        else:
            self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
                text + line[self.draw_cursor_col:]
            self.rescan_line()
        self.draw_cursor_col += len(text)
        self.cursor_col = self.draw_cursor_col

    def add_char_after_cursor(self, char: str) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = line[:self.draw_cursor_col] + \
            char + line[self.draw_cursor_col:]
        self.draw_cursor_col += 1
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def new_line(self) -> None:
        self.suggestions.clear()
        if self.draw_cursor_col != len(self.lines[self.cursor_row]):
            line = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = line[:self.draw_cursor_col]
            self.rescan_line()
            self.cursor_row += 1
            self.lines.insert(self.cursor_row, line[self.draw_cursor_col:])
            self.line_tokens.insert(self.cursor_row, [])
            self.rescan_line()
            self.cursor_col = 0
            self.draw_cursor_col = 0
            return
        self.cursor_row += 1
        self.lines.insert(self.cursor_row, "")
        self.line_tokens.insert(self.cursor_row, [])
        self.cursor_col = 0
        self.draw_cursor_col = 0

    def rescan_line(self) -> None:
        line = self.cursor_row
        self.scanner.reset(self.lines[line])
        # Expects a 1-based index
        self.scanner.line = line + 1
        self.line_tokens[line] = []
        while(token := self.scanner.scan_token()).token_type != TokenType.EOL:
            self.line_tokens[line].append(token)

    def get_text(self) -> str:
        return '\n'.join(self.lines)

    def draw_callback_px(self, context: bpy.context):
        prefs = context.preferences.addons['math_formula'].preferences
        font_id = fonts['regular']
        font_size = prefs.font_size
        font_dpi = 72
        blf.size(font_id, font_size, font_dpi)

        char_width = blf.dimensions(font_id, 'H')[0]
        char_height = blf.dimensions(font_id, 'Hq')[1]*1.3
        # Set the initial positions of the text
        posx = self.pos[0]
        posy = self.pos[1]
        posz = 0

        # Get the dimensions so that we know where to place the next stuff
        width = blf.dimensions(font_id, "Formula: ")[0]
        # Color for the non-user text.
        blf.color(font_id, 0.4, 0.5, 0.1, 1.0)
        blf.position(font_id, posx, posy+char_height, posz)
        blf.draw(
            font_id, f"(Press CTRL + ENTER to confirm, ESC to cancel)    (Line:{self.cursor_row+1} Col:{self.draw_cursor_col+1})")

        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, posx, posy, posz)
        blf.draw(font_id, "Formula: ")
        for line_num, tokens in enumerate(self.line_tokens):
            line = self.lines[line_num]
            prev = 0
            line_posx = posx+width
            line_posy = posy - char_height*line_num
            for i, token in enumerate(tokens):
                blf.position(font_id, line_posx, line_posy, posz)
                text = token.lexeme
                start = token.start
                # Draw white space
                white_space = line[prev:start]
                for char in white_space:
                    blf.position(font_id, line_posx, line_posy, posz)
                    blf.draw(font_id, char)
                    line_posx += char_width
                token_font_style = font_id
                prev_token = tokens[i-1] if i > 0 else token
                if token.token_type == TokenType.IDENTIFIER:
                    if prev_token.token_type == TokenType.COLON:
                        # Check if it's a valid type
                        color(token_font_style,
                              prefs.type_color if token.lexeme in string_to_data_type else prefs.default_color)
                    else:
                        next_token = tokens[i+1] if i + \
                            1 < len(tokens) else token
                        if next_token.token_type == TokenType.LEFT_PAREN:
                            color(token_font_style, prefs.function_color)
                        else:
                            color(token_font_style, prefs.default_color)
                elif TokenType.OUT.value <= token.token_type.value <= TokenType.AND.value:
                    color(token_font_style, prefs.keyword_color)
                elif token.token_type in (TokenType.INT, TokenType.FLOAT):
                    color(token_font_style, prefs.number_color)
                elif token.token_type == TokenType.PYTHON:
                    token_font_style = fonts['bold']
                    color(token_font_style, prefs.python_color)
                elif token.token_type == TokenType.ERROR:
                    text, error = token.lexeme
                    token_font_style = fonts['italic']
                    color(token_font_style, prefs.error_color)
                elif token.token_type == TokenType.STRING:
                    color(token_font_style, prefs.string_color)
                else:
                    color(token_font_style, prefs.default_color)
                blf.size(token_font_style, font_size, font_dpi)

                # Draw manually to ensure equal spacing and no kerning.
                for char in text:
                    blf.position(token_font_style, line_posx, line_posy, posz)
                    blf.draw(token_font_style, char)
                    line_posx += char_width
                prev = start + len(text)
            # Errors
            color(font_id, prefs.error_color)
            error_base_y = posy-char_height*(len(self.lines) + 1)
            for n, error in enumerate(self.errors):
                blf.position(font_id, posx+width,
                             error_base_y - n*char_height, posz)
                blf.draw(font_id, str(error.message))
                macro_token = error.token
                error_col = macro_token.col - 1
                error_row = macro_token.line - 1
                blf.position(font_id, posx+width+char_width *
                             error_col, posy-char_height*error_row - char_height*0.75, posz)
                blf.draw(font_id, '^'*len(error.token.lexeme))
        # Draw cursor
        blf.color(font_id, 0.1, 0.4, 0.7, 1.0)
        blf.position(font_id, posx+width+self.draw_cursor_col*char_width-char_width/2,
                     posy-char_height*self.cursor_row, posz)
        blf.draw(font_id, '|')


def color(font_id, color):
    blf.color(font_id, color[0], color[1], color[2], 1.0)