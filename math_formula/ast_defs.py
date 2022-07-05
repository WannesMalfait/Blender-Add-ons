from dataclasses import dataclass, field, fields
from typing import Union
from math_formula.scanner import Token
from math_formula.nodes.base import DataType, ValueType


@dataclass
class Ast():
    token: Token


class operator(Ast):
    ...


class And(operator):
    ...


class Or(operator):
    ...


class Add(operator):
    ...


class Div(operator):
    ...


class Mod(operator):
    ...


class Mult(operator):
    ...


class Pow(operator):
    ...


class Sub(operator):
    ...


class unaryop(Ast):
    ...


class Not(unaryop):
    ...


class USub(unaryop):
    ...


class Eq(operator):
    ...


class Gt(operator):
    ...


class GtE(operator):
    ...


class Lt(operator):
    ...


class LtE(operator):
    ...


class NotEq(operator):
    ...


class stmt(Ast):
    ...


class expr(stmt):
    ...


@dataclass
class Module(Ast):
    body: list[stmt] = field(default_factory=list)

# Literals


@dataclass
class Constant(expr):
    value: ValueType
    type: DataType


@dataclass
class Vec3(expr):
    x: Union[None, expr]
    y: Union[None, expr]
    z: Union[None, expr]


@dataclass
class Rgba(expr):
    r: Union[None, expr]
    g: Union[None, expr]
    b: Union[None, expr]
    a: Union[None, expr]


# Variables


@dataclass
class Name(expr):
    id: str

# Expressions


@dataclass
class UnaryOp(expr):
    op: unaryop
    operand: expr


@dataclass
class BinOp(expr):
    left: expr
    op: operator
    right: expr


@dataclass
class Attribute(expr):
    value: expr
    attr: str


@dataclass
class Keyword(expr):
    arg: str
    value: expr


@dataclass
class Call(expr):
    func: Union[Name, Attribute]
    pos_args: list[expr]
    keyword_args: list[Keyword]

# Statements


@dataclass
class arg(Ast):
    arg: str
    type: DataType
    default: Union[None, expr]


@dataclass
class FunctionDef(stmt):
    name: str
    args: list[arg]
    body: list[stmt]
    returns: list[arg]


@dataclass
class NodegroupDef(stmt):
    name: str
    args: list[arg]
    body: list[stmt]
    returns: list[arg]


@dataclass
class Out(stmt):
    targets: list[Union[None, Name]]
    value: expr


@dataclass
class Assign(stmt):
    targets: expr
    value: expr


@dataclass
class Loop(stmt):
    var: Union[None, Name]
    start: int
    end: int
    body: list[stmt]


# Code copied and adapted from pythons own ast module
def dump(node, indent=None):
    """
    Return a formatted dump of the tree in node.  This is mainly useful for
    debugging purposes. If indent is a non-negative
    integer or string, then the tree will be pretty-printed with that indent
    level. None (the default) selects the single line representation.
    """
    def _format(node, level=0):
        if indent is not None:
            level += 1
            prefix = '\n' + indent * level
            sep = ',\n' + indent * level
        else:
            prefix = ''
            sep = ', '
        if isinstance(node, Ast):
            cls = type(node)
            args = []
            allsimple = True
            keywords = True
            for field in fields(node):
                name = field.name
                if name == 'token':
                    continue
                try:
                    value = getattr(node, name)
                except AttributeError:
                    keywords = True
                    continue
                if value is None and getattr(cls, name, ...) is None:
                    keywords = True
                    continue
                value, simple = _format(value, level)
                allsimple = allsimple and simple
                if keywords:
                    args.append('%s=%s' % (name, value))
                else:
                    args.append(value)
            if allsimple and len(args) <= 3:
                return '%s(%s)' % (node.__class__.__name__, ', '.join(args)), not args
            return '%s(%s%s)' % (node.__class__.__name__, prefix, sep.join(args)), False
        elif isinstance(node, list):
            if not node:
                return '[]', True
            return '[%s%s]' % (prefix, sep.join(_format(x, level)[0] for x in node)), False
        return repr(node), True

    if not isinstance(node, Ast):
        raise TypeError('expected Ast, got %r' % node.__class__.__name__)
    if indent is not None and not isinstance(indent, str):
        indent = ' ' * indent
    return _format(node)[0]
