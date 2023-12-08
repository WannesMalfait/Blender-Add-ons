import os
from collections import deque

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

from . import ast_defs, file_loading
from .backends.builtin_nodes import (
    NodeInstance,
    geometry_node_aliases,
    instances,
    levenshtein_distance,
    nodes,
    shader_geo_node_aliases,
    shader_node_aliases,
)
from .backends.geometry_nodes import geometry_nodes
from .backends.shader_nodes import shader_nodes
from .backends.type_defs import string_to_data_type
from .compiler import Error
from .mf_parser import Parser
from .scanner import Scanner, Token, TokenType

add_on_dir = os.path.dirname(os.path.realpath(__file__))

font_directory = os.path.join(add_on_dir, "fonts")
fonts = {
    "bfont": 0,
    "regular": blf.load(os.path.join(font_directory, "Anonymous_Pro.ttf")),
    "italic": blf.load(os.path.join(font_directory, "Anonymous_Pro_I.ttf")),
    "bold": blf.load(os.path.join(font_directory, "Anonymous_Pro_0.ttf")),
    "bold_italic": blf.load(os.path.join(font_directory, "Anonymous_Pro_BI.ttf")),
}


rect_vertices = ((0, 0), (1, 0), (0, 1), (1, 1))

rect_indices = ((0, 1, 2), (2, 1, 3))

if not bpy.app.background:
    rect_shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    rect_batch = batch_for_shader(
        rect_shader, "TRIS", {"pos": rect_vertices}, indices=rect_indices
    )


class Editor:
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
        self.reload_fonts()

    @staticmethod
    def reload_fonts():
        global fonts
        fonts = {
            "bfont": 0,
            "regular": blf.load(os.path.join(font_directory, "Anonymous_Pro.ttf")),
            "italic": blf.load(os.path.join(font_directory, "Anonymous_Pro_I.ttf")),
            "bold": blf.load(os.path.join(font_directory, "Anonymous_Pro_0.ttf")),
            "bold_italic": blf.load(
                os.path.join(font_directory, "Anonymous_Pro_BI.ttf")
            ),
        }

    def replace_text(self, text: str):
        Editor.__init__(self, self.pos)
        self.paste_after_cursor(text)

    @staticmethod
    def get_function_output_names(
        func: NodeInstance | str, aliases: list[dict[str, NodeInstance]]
    ) -> None | list[str]:
        if isinstance(func, NodeInstance):
            node = nodes[func.key]
            return [out[0] for out in node.outputs]
        for alias_dict in aliases:
            if func in alias_dict:
                node = nodes[alias_dict[func].key]
                return [out[0] for out in node.outputs]
        return None

    def attribute_suggestions(
        self, prev_token: Token, token_under_cursor: Token, tree_type: str
    ):
        token_text = token_under_cursor.lexeme
        text_start = token_under_cursor.start
        dot_token = None
        if token_under_cursor.lexeme == ".":
            token_text = ""
            text_start += 1
            dot_token = token_under_cursor
        else:
            dot_token = prev_token

        text_start += sum(
            [len(line) + 1 for i, line in enumerate(self.lines) if i < self.cursor_row]
        )
        text = self.get_text()[:text_start]
        parser = Parser(text)
        ast = parser.parse()
        # Find where we are in the ast.
        # We know that we're always in the last statement, so we can speed up the search a little.
        node = ast_defs.find(ast.body[-1], dot_token)
        if node is None:
            return
        assert isinstance(
            node, ast_defs.Attribute
        ), "Dot token should be from attribute node"
        if not isinstance(node.value, ast_defs.Call):
            return
        func = node.value.func
        func_name = ""
        if isinstance(func, ast_defs.Attribute):
            func_name = func.attr
        else:
            func_name = func.id
        # Find which node this belongs to.
        suggestions = None
        if func_name in instances:
            suggestions = self.get_function_output_names(
                instances[func_name][0], [shader_geo_node_aliases]
            )
        elif tree_type == "GeometryNodeTree":
            if func_name in geometry_nodes:
                suggestions = self.get_function_output_names(
                    geometry_nodes[func_name][0],
                    [shader_geo_node_aliases, geometry_node_aliases],
                )
            else:
                suggestions = self.get_function_output_names(
                    func_name, [shader_geo_node_aliases, geometry_node_aliases]
                )
        elif tree_type == "ShaderNodeTree":
            if func_name in shader_nodes:
                suggestions = self.get_function_output_names(
                    shader_nodes[func_name][0],
                    [shader_geo_node_aliases, shader_node_aliases],
                )
            else:
                suggestions = self.get_function_output_names(
                    func_name, [shader_geo_node_aliases, shader_node_aliases]
                )
        if suggestions is not None:
            self.suggestions += [
                name for name in suggestions if name.startswith(token_text)
            ]
            return
        ty_func = None
        if (
            tree_type == "GeometryNodeTree"
            and func_name in file_loading.file_data.geometry_nodes
        ):
            ty_func = file_loading.file_data.geometry_nodes[func_name][0]
        elif (
            tree_type == "ShaderNodeTree"
            and func_name in file_loading.file_data.shader_nodes
        ):
            ty_func = file_loading.file_data.shader_nodes[func_name][0]
        if ty_func is not None:
            self.suggestions += [
                out.name for out in ty_func.outputs if out.name.startswith(token_text)
            ]

    def token_under_cursor(self) -> tuple[None | Token, None | Token]:
        """Returns the token under the cursor, and the token before that"""
        prev_token = None
        for token in self.line_tokens[self.cursor_row]:
            if token.start < self.draw_cursor_col <= token.start + len(token.lexeme):
                return token, prev_token
            prev_token = token
        return (None, None)

    def try_auto_complete(self, tree_type: str) -> None:
        token_under_cursor, prev_token = self.token_under_cursor()
        if token_under_cursor is None:
            return
        if len(self.suggestions) != 0:
            # Already calculated suggestions, so just use those.
            suggestion = self.suggestions.popleft()
            token_to_replace = token_under_cursor
            if token_under_cursor.token_type is TokenType.LEFT_PAREN:
                # Last thing we completed was a function, so we have
                # an extra "()" at the end.
                assert (
                    prev_token is not None
                ), "We completed a function, there should be a token before the '('"
                token_to_replace = prev_token
                # Remove the extra "()"
                self.lines[self.cursor_row] = (
                    self.lines[self.cursor_row][: self.cursor_col - 1]
                    + self.lines[self.cursor_row][self.cursor_col + 1 :]
                )
            if " " in suggestion:
                self.replace_token(token_to_replace, f"n'{suggestion}'")
            else:
                self.replace_token(token_to_replace, suggestion)

            if suggestion.endswith("()"):
                # Place the cursor between the parenthesis.
                self.draw_cursor_col -= 1
                self.cursor_col = self.draw_cursor_col
            self.suggestions.append(suggestion)
            return
        if prev_token is not None:
            if prev_token.lexeme == "." or token_under_cursor.lexeme == ".":
                self.attribute_suggestions(prev_token, token_under_cursor, tree_type)
        options = list(instances.keys())
        if tree_type == "GeometryNodeTree":
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
                self.suggestions.append(name + "()")
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
                    options_with_dist.append((option + "()", d))
            sorted_options = sorted(options_with_dist, key=lambda x: x[1])
            self.suggestions += list(map(lambda x: x[0], sorted_options))
        else:
            self.suggestions = deque(sorted(self.suggestions, key=len))
        if len(self.suggestions) == 0:
            return
        suggestion = self.suggestions.popleft()
        if token_under_cursor.lexeme == ".":
            self.text_after_cursor(suggestion)
        else:
            self.replace_token(token_under_cursor, suggestion)

        if suggestion.endswith("()"):
            # Place the cursor between the parenthesis.
            self.draw_cursor_col -= 1
            self.cursor_col = self.draw_cursor_col
        self.suggestions.append(suggestion)

    def text_after_cursor(self, text: str) -> None:
        line = self.lines[self.cursor_row]
        self.lines[self.cursor_row] = (
            line[: self.draw_cursor_col] + text + line[self.draw_cursor_col :]
        )
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
                len(self.lines[self.cursor_row]), self.cursor_col
            )

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
                len(self.lines[self.cursor_row]), self.cursor_col
            )

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
            self.lines[self.cursor_row - 1] += self.lines[self.cursor_row]
            self.cursor_row -= 1
            self.rescan_line()
            self.cursor_col = self.draw_cursor_col
            self.lines.pop(self.cursor_row + 1)
            self.line_tokens.pop(self.cursor_row + 1)
            return

        line = self.lines[self.cursor_row]
        match (self.get_char_before_cursor(), self.get_char_after_cursor()):
            case ("(", ")") | ("{", "}") | ("[", "]") | ('"', '"') | ("'", "'"):
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col - 1] + line[self.draw_cursor_col + 1 :]
                )
            case _:
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col - 1] + line[self.draw_cursor_col :]
                )
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
            self.lines.pop(self.cursor_row + 1)
            self.line_tokens.pop(self.cursor_row + 1)
            return
        self.draw_cursor_col += 1
        match (self.get_char_before_cursor(), self.get_char_after_cursor()):
            case ("(", ")") | ("{", "}") | ("[", "]") | ('"', '"') | ("'", "'"):
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col - 1] + line[self.draw_cursor_col + 1 :]
                )
            case _:
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col - 1] + line[self.draw_cursor_col :]
                )
        self.draw_cursor_col -= 1
        self.rescan_line()

    def paste_after_cursor(self, text: str) -> None:
        # Replace tabs with two spaces since the font drawing code doesn't like tabs.
        text = text.expandtabs(2)
        self.suggestions.clear()
        line = self.lines[self.cursor_row]
        if (index := text.find("\n")) != -1:
            self.lines[self.cursor_row] = line[: self.draw_cursor_col] + text[:index]
            self.rescan_line()
            line = line[self.draw_cursor_col :]
            text = text[index + 1 :]
            self.draw_cursor_col = len(self.lines[self.cursor_row])
            self.new_line()
            while True:
                if text == "":
                    break
                if (index := text.find("\n")) != -1:
                    self.lines[self.cursor_row] = text[:index]
                    self.rescan_line()
                    text = text[index + 1 :]
                    self.draw_cursor_col = len(self.lines[self.cursor_row])
                    self.new_line()
                else:
                    self.lines[self.cursor_row] = text + line
                    self.draw_cursor_col = 0
                    self.rescan_line()
                    break
        else:
            self.lines[self.cursor_row] = (
                line[: self.draw_cursor_col] + text + line[self.draw_cursor_col :]
            )
            self.rescan_line()
        self.draw_cursor_col += len(text)
        self.cursor_col = self.draw_cursor_col

    def add_char_after_cursor(self, char: str) -> None:
        self.suggestions.clear()
        line = self.lines[self.cursor_row]

        # Auto-close parenthesis.
        match char:
            case "(":
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col] + "()" + line[self.draw_cursor_col :]
                )
            case "[":
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col] + "[]" + line[self.draw_cursor_col :]
                )
            case "{":
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col] + "{}" + line[self.draw_cursor_col :]
                )
            # Only add closing bracket if not already there
            case ")" | "]" | "}":
                if self.get_char_after_cursor() != char:
                    self.lines[self.cursor_row] = (
                        line[: self.draw_cursor_col]
                        + char
                        + line[self.draw_cursor_col :]
                    )
            case '"' | "'":
                text = char * 2
                if (token := self.token_under_cursor()[0]) is not None:
                    if token.token_type is TokenType.ERROR:
                        # Possibly inside an unclosed string,
                        # so just add the closing " or '
                        text = char
                    if self.get_char_after_cursor() == char:
                        text = ""
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col] + text + line[self.draw_cursor_col :]
                )
            case _:
                self.lines[self.cursor_row] = (
                    line[: self.draw_cursor_col] + char + line[self.draw_cursor_col :]
                )
        self.draw_cursor_col += 1
        self.cursor_col = self.draw_cursor_col
        self.rescan_line()

    def indentation(self, row: int) -> int:
        """The number of spaces at the start of the given line"""
        line = self.lines[row]
        return len(line) - len(line.lstrip())

    def get_char_before_cursor(self) -> str | None:
        if self.draw_cursor_col <= 0 or self.draw_cursor_col - 1 >= len(
            self.lines[self.cursor_row]
        ):
            return None

        return self.lines[self.cursor_row][self.draw_cursor_col - 1]

    def get_char_after_cursor(self) -> str | None:
        if len(self.lines[self.cursor_row]) == 0 or self.draw_cursor_col >= len(
            self.lines[self.cursor_row]
        ):
            return None

        return self.lines[self.cursor_row][self.draw_cursor_col]

    def new_line(self) -> None:
        self.suggestions.clear()
        indentation = self.indentation(self.cursor_row)
        if self.get_char_before_cursor() == "{":
            indentation += 2
        if self.draw_cursor_col != len(self.lines[self.cursor_row]):
            closing_brace = self.get_char_after_cursor() == "}"
            line = self.lines[self.cursor_row]
            self.lines[self.cursor_row] = line[: self.draw_cursor_col]
            self.rescan_line()
            self.cursor_row += 1
            if closing_brace:
                self.lines.insert(self.cursor_row, " " * indentation)
                self.line_tokens.insert(self.cursor_row, [])
                self.rescan_line()
                self.cursor_row += 1
                self.lines.insert(
                    self.cursor_row,
                    " " * max(0, indentation - 2) + line[self.draw_cursor_col :],
                )
                self.line_tokens.insert(self.cursor_row, [])
                self.rescan_line()
                self.cursor_row -= 1
            else:
                self.lines.insert(
                    self.cursor_row, " " * indentation + line[self.draw_cursor_col :]
                )
                self.line_tokens.insert(self.cursor_row, [])
                self.rescan_line()
            self.cursor_col = indentation
            self.draw_cursor_col = indentation
            return
        self.cursor_row += 1
        self.lines.insert(self.cursor_row, " " * indentation)
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
        return "\n".join(self.lines)

    def draw_suggestions(
        self,
        formula_pos: tuple[float, float],
        padding: float,
        char_size: tuple[float, float],
        num_suggestions: int,
        bg_color: tuple[float, float, float],
        font_color: tuple[float, float, float],
        font_id: int,
    ):
        # First calculate where the suggestion will appear
        posx, posy = formula_pos
        char_width, char_height = char_size
        posy -= self.cursor_row * char_height
        suggestion_offset = char_width * self.draw_cursor_col
        token_uc, prev_token_uc = self.token_under_cursor()
        if token_uc is not None and token_uc.token_type is TokenType.LEFT_PAREN:
            assert prev_token_uc is not None, "Should be the function identifier"
            suggestion_offset -= char_width * (len(prev_token_uc.lexeme) + 1)
        elif token_uc is not None and token_uc.token_type is TokenType.IDENTIFIER:
            suggestion_offset -= char_width * (len(token_uc.lexeme))

        gpu.matrix.translate(
            [
                posx + suggestion_offset - padding,
                posy - 0.3 * char_height,
            ]
        )
        gpu.matrix.scale(
            [
                max(
                    [
                        len(suggestion)
                        for _, suggestion in zip(
                            range(num_suggestions), self.suggestions
                        )
                    ]
                )
                * char_width
                + 2 * padding,
                -min(num_suggestions + 1, len(self.suggestions)) * char_height
                - padding,
            ]
        )
        rect_shader.uniform_float("color", (bg_color[0], bg_color[1], bg_color[2], 1))
        rect_batch.draw(rect_shader)
        gpu.matrix.load_identity()

        # Show the suggestions.
        for i, suggestion in zip(range(num_suggestions + 1), self.suggestions):
            # Decrease alpha with each suggestion.
            blf.color(
                font_id,
                font_color[0],
                font_color[1],
                font_color[2],
                max(0.6 ** (i + 1), 0.03),
            )
            blf.position(
                font_id,
                posx + suggestion_offset,
                posy - (i + 1) * char_height,
                0,
            )
            if i == num_suggestions:
                # Just show that there are more suggestions
                suggestion = "..."
            blf.draw(font_id, suggestion)

    def draw_callback_px(self, context: bpy.types.Context):
        prefs = context.preferences.addons["math_formula"].preferences

        font_id = fonts["regular"]
        font_size = prefs.font_size  # type: ignore
        blf.size(font_id, font_size)

        char_width = blf.dimensions(font_id, "H")[0]
        char_height = blf.dimensions(font_id, "Hq")[1] * 1.3
        # Set the initial positions of the text
        posx = self.pos[0]
        posy = self.pos[1]
        posz = 0

        # Get the dimensions so that we know where to place the next stuff
        formula_width: float = blf.dimensions(font_id, "Formula: ")[0]
        info_message = (
            "(Press CTRL + ENTER to confirm, ESC to cancel)"
            + f"    (Line:{self.cursor_row+1} Col:{self.draw_cursor_col+1})"
        )
        info_width = blf.dimensions(font_id, info_message)[0]

        # Bounding box for draw area
        bb_width = max(
            max([len(line) for line in self.lines]) * char_width + formula_width,
            info_width,
        )
        bb_height = (len(self.lines) + 1) * char_height

        # Draw the background
        gpu.state.blend_set("ALPHA")
        padding = 20
        gpu.matrix.translate([posx - padding, posy + 2 * char_height + padding])
        gpu.matrix.scale([bb_width + 2 * padding, -bb_height - 2 * padding])
        bg = prefs.background_color  # type: ignore
        alpha = prefs.background_alpha  # type: ignore
        rect_shader.uniform_float("color", (bg[0], bg[1], bg[2], alpha))
        rect_batch.draw(rect_shader)
        gpu.matrix.load_identity()
        gpu.state.blend_set("NONE")

        # Draw the info message
        blf.color(font_id, 0.4, 0.5, 0.1, 1.0)
        blf.position(font_id, posx, posy + char_height, posz)
        blf.draw(font_id, info_message)

        blf.color(font_id, 1.0, 1.0, 1.0, 1.0)
        blf.position(font_id, posx, posy, posz)
        blf.draw(font_id, "Formula: ")
        for line_num, tokens in enumerate(self.line_tokens):
            line = self.lines[line_num]
            prev = 0
            line_posx = posx + formula_width
            line_posy = posy - char_height * line_num
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
                prev_token = tokens[i - 1] if i > 0 else token
                if token.token_type == TokenType.IDENTIFIER:
                    if prev_token.token_type == TokenType.COLON:
                        # Check if it's a valid type
                        color(
                            token_font_style,
                            prefs.type_color  # type: ignore
                            if token.lexeme in string_to_data_type
                            else prefs.identifier_color,  # type: ignore
                        )
                    else:
                        next_token = tokens[i + 1] if i + 1 < len(tokens) else token
                        if next_token.token_type == TokenType.LEFT_PAREN:
                            color(
                                token_font_style, prefs.function_color  # type: ignore
                            )
                        else:
                            color(token_font_style, prefs.identifier_color)  # type: ignore
                elif (
                    TokenType.OUT.value <= token.token_type.value <= TokenType.AND.value
                ):
                    color(token_font_style, prefs.keyword_color)  # type: ignore
                elif token.token_type in (TokenType.INT, TokenType.FLOAT):
                    color(token_font_style, prefs.number_color)  # type: ignore
                elif token.token_type == TokenType.PYTHON:
                    token_font_style = fonts["bold"]
                    color(token_font_style, prefs.python_color)  # type: ignore
                elif token.token_type == TokenType.ERROR:
                    token_font_style = fonts["italic"]
                    color(token_font_style, prefs.error_color)  # type: ignore
                elif token.token_type == TokenType.STRING:
                    color(token_font_style, prefs.string_color)  # type: ignore
                elif token.token_type == TokenType.GROUP_NAME:
                    color(token_font_style, prefs.function_color)  # type: ignore
                else:
                    color(token_font_style, prefs.default_color)  # type: ignore
                blf.size(token_font_style, font_size)

                # Draw manually to ensure equal spacing and no kerning.
                for char in text:
                    blf.position(token_font_style, line_posx, line_posy, posz)
                    blf.draw(token_font_style, char)
                    line_posx += char_width
                prev = start + len(text)
            # Errors
            color(font_id, prefs.error_color)  # type: ignore
            error_base_y = posy - char_height * (len(self.lines) + 1)
            for n, error in enumerate(self.errors):
                blf.position(
                    font_id, posx + formula_width, error_base_y - n * char_height, posz
                )
                blf.draw(font_id, str(error.message))
                macro_token = error.token
                error_col = macro_token.col - 1
                error_row = macro_token.line - 1
                blf.position(
                    font_id,
                    posx + formula_width + char_width * error_col,
                    posy - char_height * error_row - char_height * 0.75,
                    posz,
                )
                blf.draw(font_id, "^" * len(error.token.lexeme))

        # Draw cursor
        blf.color(font_id, 0.1, 0.4, 0.7, 1.0)
        blf.position(
            font_id,
            posx + formula_width + self.draw_cursor_col * char_width - char_width / 2,
            posy - char_height * self.cursor_row,
            posz,
        )
        blf.draw(font_id, "|")

        # Show suggestions.
        # Needs to be done after drawing text,
        # since the suggestions appear above text.
        # Only show if we have more than one suggestion.
        if len(self.suggestions) > 1:
            self.draw_suggestions(
                (posx + formula_width, posy),
                10,
                (char_width, char_height),
                num_suggestions=10,
                bg_color=prefs.background_color,  # type: ignore
                font_color=prefs.default_color,  # type: ignore
                font_id=font_id,
            )


def color(font_id, color):
    blf.color(font_id, color[0], color[1], color[2], 1.0)
