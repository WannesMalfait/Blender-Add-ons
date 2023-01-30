import os
import blf
import bpy
from collections import deque
from . import file_loading
from . import ast_defs
from .mf_parser import Parser
from .scanner import Scanner, Token, TokenType
from .compiler import Error
from .backends.main import string_to_data_type
from .backends.builtin_nodes import instances, levenshtein_distance, nodes, shader_node_aliases, shader_geo_node_aliases, geometry_node_aliases, NodeInstance
from .backends.geometry_nodes import geometry_nodes
from .backends.shader_nodes import shader_nodes


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

    def replace_text(self, text: str):
        self.__init__(self.pos)
        self.paste_after_cursor(text)

    @staticmethod
    def get_function_output_names(func: NodeInstance | str, aliases: list[dict[str, NodeInstance]]) -> None | list[str]:
        if isinstance(func, NodeInstance):
            node = nodes[func.key]
            return [out[0] for out in node.outputs]
        for alias_dict in aliases:
            if func in alias_dict:
                node = nodes[alias_dict[func].key]
                return [out[0] for out in node.outputs]

    def attribute_suggestions(self, prev_token: Token, token_under_cursor: Token, tree_type: str):
        token_text = token_under_cursor.lexeme
        text_start = token_under_cursor.start
        dot_token = None
        if token_under_cursor.lexeme == '.':
            token_text = ''
            text_start += 1
            dot_token = token_under_cursor
        else:
            dot_token = prev_token

        text_start += sum([len(line) + 1
                           for i, line in enumerate(self.lines) if i < self.cursor_row])
        text = self.get_text()[:text_start]
        parser = Parser(text)
        ast = parser.parse()
        # Find where we are in the ast.
        # We know that we're always in the last statement, so we can speed up the search a little.
        node = ast_defs.find(ast.body[-1], dot_token)
        if node is None:
            return
        assert isinstance(
            node, ast_defs.Attribute), 'Dot token should be from attribute node'
        if not isinstance(node.value, ast_defs.Call):
            return
        func = node.value.func
        func_name = ''
        if isinstance(func, ast_defs.Attribute):
            func_name = func.attr
        else:
            func_name = func.id
        # Find which node this belongs to.
        suggestions = None
        if func_name in instances:
            suggestions = self.get_function_output_names(
                instances[func_name][0], [shader_geo_node_aliases])
        elif tree_type == 'GeometryNodeTree':
            if func_name in geometry_nodes:
                suggestions = self.get_function_output_names(
                    geometry_nodes[func_name][0], [shader_geo_node_aliases, geometry_node_aliases])
            else:
                suggestions = self.get_function_output_names(
                    func_name, [shader_geo_node_aliases, geometry_node_aliases])
        elif tree_type == 'ShaderNodeTree':
            if func_name in shader_nodes:
                suggestions = self.get_function_output_names(
                    shader_nodes[func_name][0], [shader_geo_node_aliases, shader_node_aliases])
            else:
                suggestions = self.get_function_output_names(
                    func_name, [shader_geo_node_aliases, shader_node_aliases])
        if suggestions is not None:
            self.suggestions += [
                name for name in suggestions if name.startswith(token_text)]
            return
        ty_func = None
        if tree_type == 'GeometryNodeTree' and func_name in file_loading.file_data.geometry_nodes:
            ty_func = file_loading.file_data.geometry_nodes[func_name][0]
        elif tree_type == 'ShaderNodeTree' and func_name in file_loading.file_data.shader_nodes:
            ty_func = file_loading.file_data.shader_nodes[func_name][0]
        if ty_func is not None:
            self.suggestions += [
                out.name for out in ty_func.outputs if out.name.startswith(token_text)]

    def try_auto_complete(self, tree_type: str) -> None:
        token_under_cursor = None
        prev_token = None
        for token in self.line_tokens[self.cursor_row]:
            if token.start < self.draw_cursor_col <= token.start + len(token.lexeme):
                token_under_cursor = token
                break
            prev_token = token
        if token_under_cursor is not None:
            if len(self.suggestions) != 0:
                # Already calculated suggestions, so just use those.
                suggestion = self.suggestions.popleft()
                if ' ' in suggestion:
                    self.replace_token(token_under_cursor, f"n'{suggestion}'")
                else:
                    self.replace_token(token_under_cursor, suggestion)
                self.suggestions.append(suggestion)
                return
            if prev_token is not None:
                if prev_token.lexeme == '.' or token_under_cursor.lexeme == '.':
                    self.attribute_suggestions(
                        prev_token, token_under_cursor, tree_type)
            options = list(instances.keys())
            if tree_type == 'GeometryNodeTree':
                options += list(geometry_nodes.keys())
                options += list(file_loading.file_data.geometry_nodes.keys())
                options += list(shader_geo_node_aliases.keys())
                options += list(geometry_node_aliases.keys())
            else:
                options += list(shader_nodes.keys())
                options += list(file_loading.file_data.shader_nodes.keys())
                options += list(shader_geo_node_aliases.keys())
                options += list(shader_node_aliases.keys())
            for name in options:
                if name.startswith(token_under_cursor.lexeme):
                    self.suggestions.append(name)
            if len(self.suggestions) == 0:
                # No exact matches, try with Levensthein distance
                # Only do this if we have at least some text
                if len(token_under_cursor.lexeme) < 4:
                    return
                options_with_dist = []
                for option in options:
                    d = levenshtein_distance(option, token_under_cursor.lexeme)
                    # Only add the best options
                    if d < 5:
                        options_with_dist.append((option, d))
                sorted_options = sorted(options_with_dist,
                                        key=lambda x: x[1])
                self.suggestions += list(map(lambda x: x[0], sorted_options))
            else:
                self.suggestions = deque(sorted(self.suggestions, key=len))
            if len(self.suggestions) == 0:
                return
            suggestion = self.suggestions.popleft()
            if token_under_cursor.lexeme == '.':
                self.text_after_cursor(suggestion)
            else:
                self.replace_token(token_under_cursor, suggestion)
            self.suggestions.append(suggestion)

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
        # Replace tabs with two spaces since the font drawing code doesn't like tabs.
        text.replace('\t', '  ')
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

    def indentation(self, row: int) -> int:
        """The number of spaces at the start of the given line"""
        line = self.lines[row]
        return len(line) - len(line.lstrip())

    def get_char_before_cursor(self) -> str | None:
        if self.cursor_col - 1 >= len(self.lines[self.cursor_row]):
            return None
        return self.lines[self.cursor_row][self.cursor_col - 1]

    def new_line(self) -> None:
        self.suggestions.clear()
        indentation = self.indentation(self.cursor_row)
        if self.get_char_before_cursor() == '{':
            indentation += 2
        if self.draw_cursor_col != len(self.lines[self.cursor_row]):
            line = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = line[:self.draw_cursor_col]
            self.rescan_line()
            self.cursor_row += 1
            self.lines.insert(self.cursor_row, ' ' *
                              indentation + line[self.draw_cursor_col:])
            self.line_tokens.insert(self.cursor_row, [])
            self.rescan_line()
            self.cursor_col = indentation
            self.draw_cursor_col = indentation
            return
        self.cursor_row += 1
        self.lines.insert(self.cursor_row, ' '*indentation)
        self.line_tokens.insert(self.cursor_row, [])
        self.cursor_col = indentation
        self.draw_cursor_col = indentation

    def rescan_line(self) -> None:
        line = self.cursor_row
        self.scanner.reset(self.lines[line])
        # Expects a 1-based index
        self.scanner.line = line + 1
        self.line_tokens[line] = []
        while (token := self.scanner.scan_token()).token_type != TokenType.EOL:
            self.line_tokens[line].append(token)

    def get_text(self) -> str:
        return '\n'.join(self.lines)

    def draw_callback_px(self, context: bpy.types.Context):
        prefs = context.preferences.addons['math_formula'].preferences
        font_id = fonts['regular']
        font_size = prefs.font_size  # type: ignore
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
                              prefs.type_color if token.lexeme in string_to_data_type else prefs.default_color)  # type: ignore
                    else:
                        next_token = tokens[i+1] if i + \
                            1 < len(tokens) else token
                        if next_token.token_type == TokenType.LEFT_PAREN:
                            color(token_font_style,
                                  prefs.function_color)  # type: ignore
                        else:
                            color(token_font_style,
                                  prefs.default_color)  # type: ignore
                elif TokenType.OUT.value <= token.token_type.value <= TokenType.AND.value:
                    color(token_font_style, prefs.keyword_color)  # type: ignore
                elif token.token_type in (TokenType.INT, TokenType.FLOAT):
                    color(token_font_style, prefs.number_color)  # type: ignore
                elif token.token_type == TokenType.PYTHON:
                    token_font_style = fonts['bold']
                    color(token_font_style, prefs.python_color)  # type: ignore
                elif token.token_type == TokenType.ERROR:
                    token_font_style = fonts['italic']
                    color(token_font_style, prefs.error_color)  # type: ignore
                elif token.token_type == TokenType.STRING:
                    color(token_font_style, prefs.string_color)  # type: ignore
                elif token.token_type == TokenType.GROUP_NAME:
                    color(token_font_style,
                          prefs.function_color)  # type: ignore
                else:
                    color(token_font_style, prefs.default_color)  # type: ignore
                blf.size(token_font_style, font_size, font_dpi)

                # Draw manually to ensure equal spacing and no kerning.
                for char in text:
                    blf.position(token_font_style, line_posx, line_posy, posz)
                    blf.draw(token_font_style, char)
                    line_posx += char_width
                prev = start + len(text)
            # Errors
            color(font_id, prefs.error_color)  # type: ignore
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
