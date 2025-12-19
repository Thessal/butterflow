from typing import List, Dict

# ==========================================
# PHASE 1: Type System & Schema Definition
# ==========================================


class Type:
    def matches(self, other: 'Type') -> bool:
        raise NotImplementedError


class AnyType(Type):
    def matches(self, other): return True
    def __repr__(self): return "Any"


class Atomic(Type):
    def __init__(self, name): self.name = name
    def matches(self, other): return isinstance(
        other, Atomic) and self.name == other.name

    def __repr__(self): return self.name


class Generic(Type):
    def __init__(self, base, inner):
        self.base = base
        self.inner = inner

    def matches(self, other):
        return isinstance(other, Generic) and self.base == other.base and self.inner.matches(other.inner)

    def __repr__(self): return f"{self.base}<{self.inner}>"


class Either(Type):
    def __init__(self, options: List[Type]): self.options = options

    def matches(self, other):
        return any(opt.matches(other) for opt in self.options)

    def __repr__(self): return f"Either({self.options})"


class DictType(Type):
    def __init__(self, fields: Dict[str, Type]): self.fields = fields

    def matches(self, other):
        if not isinstance(other, DictType):
            return False
        # Check if all required fields in self exist in other and match types
        for k, t in self.fields.items():
            if k not in other.fields:
                return False
            if not t.matches(other.fields[k]):
                return False
        return True

    def __repr__(self): return f"Dict({self.fields})"


class TupleType(Type):
    def __init__(self, items: List[Type]): self.items = items

    def matches(self, other):
        if not isinstance(other, TupleType) or len(self.items) != len(other.items):
            return False
        return all(s.matches(o) for s, o in zip(self.items, other.items))

    def __repr__(self): return f"Tuple({self.items})"

# The "Operator" type represents a function signature: Operator<Args, ReturnType>


class Operator(Type):
    def __init__(self, args: Type, ret: Type):
        self.args = args  # Can be DictType or TupleType
        self.ret = ret

    def matches(self, other):
        # An Operator matches another if their return types match
        # (Covariance usually, but strict equality for now)
        if not isinstance(other, Operator):
            return False
        return self.ret.matches(other.ret)

    def __repr__(self): return f"Operator<{self.args}, {self.ret}>"
