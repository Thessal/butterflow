import re
from typing import List
from .typesystem import Atomic, Generic

# ==========================================
# PHASE 2: Lexer & Parser (AST Generation)
# ==========================================


class Token:
    def __init__(self, type_, value):
        self.type: str = type_
        self.value: str = value

    def __repr__(self): return f"Tok({self.type}, {self.value})"


def lex(text) -> List[Token]:
    specs = [
        ('COMMENT', r'#.*'),
        ('FLOAT',  r'\d+\.\d*'),
        ('INT',    r'\d+'),
        ('STRING', r'"[^"]*"'),
        ('BOOL',   r'\b(True|False)\b'),  # Added Bool
        ('ID',     r'[A-Za-z_][A-Za-z0-9_]*'),
        ('OP',     r'[:=(){}<>,]'),
        ('SKIP',   r'[ \t\n]+'),
    ]
    tok_regex = '|'.join('(?P<%s>%s)' % pair for pair in specs)
    tokens = []
    for mo in re.finditer(tok_regex, text):
        kind = mo.lastgroup
        val = mo.group()
        if kind == 'COMMENT':
            continue
        if kind == 'SKIP':
            continue
        if kind == 'STRING':
            val = val.strip('"')
        if kind == 'BOOL':
            val = (val == 'True')  # Convert to python bool
        tokens.append(Token(kind, val))
    return tokens

# AST Nodes


class Expr:
    pass


class Literal(Expr):
    def __init__(self, val, type_): self.val, self.type_ = val, type_
    def __repr__(self): return f"{self.val}:{self.type_}"


class VarRef(Expr):
    def __init__(self, name): self.name = name
    def __repr__(self): return self.name


class Call(Expr):
    def __init__(self, func, args): self.func, self.args = func, args
    def __repr__(self): return f"{self.func}({self.args})"


class Block(Expr):
    def __init__(self, assigns): self.assigns = assigns


class Stmt:
    pass


class Assign(Stmt):
    def __init__(self, target, expr): self.target, self.expr = target, expr


class FuncDef(Stmt):
    def __init__(self, name, args, return_type,
                 body): self.name, self.args, self.return_type, self.body = name, args, return_type, body


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def consume(self, type_=None, val=None) -> Token:
        if self.pos >= len(self.tokens):
            raise Exception("Unexpected EOF")
        t = self.tokens[self.pos]
        if type_ and t.type != type_:
            raise Exception(f"Expected {type_}, got {t.type}")
        if val and t.value != val:
            raise Exception(f"Expected {val}, got {t.value}")
        self.pos += 1
        return t

    def parse_type(self) -> Atomic | Generic:
        base: str = self.consume('ID').value
        if self.pos < len(self.tokens) and self.tokens[self.pos].value == '<':
            self.consume(val='<')
            inner = self.parse_type()
            self.consume(val='>')
            return Generic(base, inner)
        return Atomic(base)

    def parse_expr(self):
        t = self.tokens[self.pos]
        if t.type == 'FLOAT':
            return Literal(float(self.consume().value), Atomic("Float"))
        if t.type == 'INT':
            return Literal(int(self.consume().value), Atomic("Int"))
        if t.type == 'BOOL':
            return Literal(self.consume().value, Atomic("Bool"))
        if t.type == 'STRING':
            return Literal(self.consume().value, Atomic("String"))
        if t.value == '{':
            return self.parse_block()

        if t.type == 'ID':
            name = self.consume().value
            if self.pos < len(self.tokens) and self.tokens[self.pos].value == '(':
                return self.parse_call(name)
            return VarRef(name)
        raise Exception(f"Unknown expr {t}")

    def parse_call(self, name):
        self.consume(val='(')
        args = {}
        idx = 0
        while self.tokens[self.pos].value != ')':
            # Check for kwarg
            if self.tokens[self.pos].type == 'ID' and self.tokens[self.pos+1].value == '=':
                k = self.consume().value
                self.consume(val='=')
                args[k] = self.parse_expr()
            else:
                args[str(idx)] = self.parse_expr()
                idx += 1
            if self.tokens[self.pos].value == ',':
                self.consume()
        self.consume(val=')')
        return Call(name, args)

    def parse_block(self):
        self.consume(val='{')
        assigns = {}
        while self.tokens[self.pos].value != '}':
            k = self.consume('ID').value
            self.consume(val='=')
            assigns[k] = self.parse_expr()
            if self.tokens[self.pos].value == ',':
                self.consume()
        self.consume(val='}')
        return Block(assigns)

    def parse(self):
        stmts = []
        names = set()
        while self.pos < len(self.tokens):
            self.parse_one(stmts, names)
        return stmts

    def parse_one(self, stmts, names):
        # Lookahead for FuncDef: ID ( ... ) =
        is_func = False
        start = self.pos
        try:
            if self.tokens[start].type == 'ID' and self.tokens[start+1].value == '(':
                is_func = True
        except:
            pass

        if is_func:
            name = self.consume('ID').value
            self.consume(val='(')

            # Parse Typed Arguments
            args = {}
            while self.tokens[self.pos].value != ')':
                arg_name = self.consume('ID').value
                self.consume(val=':')  # Enforce type annotation
                arg_type = self.parse_type()  # Assumes a parse_type helper exists
                args[arg_name] = arg_type

                if self.tokens[self.pos].value == ',':
                    self.consume()
            self.consume(val=')')

            # Parse Return Type
            self.consume(val=':')
            return_type = self.parse_type()

            self.consume(val='=')

            # Handle Block Scope
            if self.tokens[self.pos].value == '{':
                self.consume(val='{')
                body_stmts: List[Stmt] = []
                body_names = set()
                # Parse statements until closing brace
                while self.tokens[self.pos].value != '}':
                    # recursively call parse() or parse_stmt() depending on your grammar structure
                    # For this snippet, assuming parse_expr or specific stmt logic:
                    self.parse_one(body_stmts, body_names)
                self.consume(val='}')
                # Wrap stmts in a Block node
                body = Block(body_stmts)
            else:
                body = self.parse_expr()

            if name in names:
                raise Exception(f"Function {name} is duplicate")
            stmts.append(FuncDef(name, args, return_type, body))
            names.add(name)
        else:
            name = self.consume('ID').value
            type_ann = None
            if self.tokens[self.pos].value == ':':
                self.consume()
                type_ann = self.parse_type()
            else:
                type_ann = Generic("Signal", "Float")
            target = (name, type_ann)
            self.consume(val='=')
            expr = self.parse_expr()
            if name in names:
                debug_msg = ""
                debug_msg += "\n[Known names]"
                for x in stmts:
                    debug_msg += repr((x, x.__dict__))
                debug_msg += "\n[Duplicate]"
                debug_msg += repr((target, expr))
                debug_msg += "\n"
                raise Exception(f"Variable {name} is duplicate\n{debug_msg}")
            stmts.append(Assign(target, expr))
            names.add(name)
