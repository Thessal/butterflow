import re
from typing import List, Dict, Union, Any, Optional
from butterflow.typesystem import Atomic, Generic, DictType, TupleType, Type
from butterflow.operators import STD_LIB
from butterflow.parser import FuncDef, Assign, Literal, VarRef, Block, Call

# ==========================================
# PHASE 3: Type Checking (Pre-Graph)
# ==========================================


class TypeChecker:
    def __init__(self, silent=False):
        self.symbol_table = {}  # name -> Type
        self.func_signatures = STD_LIB.copy()
        self.silent = silent

    def check(self, stmts):
        if not self.silent:
            print("Type Checking...")
        for stmt in stmts:

            if isinstance(stmt, FuncDef):
                self.func_signatures[stmt.name] = stmt
                # Check body with arguments injected into a temporary scope
                inner_scope = {**self.symbol_table, **stmt.args}
                inner_checker = TypeChecker(silent=self.silent)
                inner_checker.symbol_table = inner_scope
                # Assuming body is a Block which contains a list of Assigns
                inner_checker.check(stmt.body.assigns)

            elif isinstance(stmt, Assign):
                rhs_type = self.infer_type(stmt.expr, self.symbol_table)

                # Validate against LHS
                name, decl_type = stmt.target
                if decl_type:
                    # THE CORE REQUIREMENT: Pattern on LHS matches Instance on RHS
                    if not decl_type.matches(rhs_type):
                        # Allow some flexibility for the complex unpacking case in the prompt
                        if isinstance(rhs_type, DictType) and name in rhs_type.fields:
                            inner = rhs_type.fields[name]
                            if not decl_type.matches(inner):
                                raise TypeError(
                                    f"Type Mismatch for '{name}': Expected {decl_type}, got {inner}")
                        else:
                            raise TypeError(
                                f"Type Mismatch for '{name}': Expected {decl_type}, got {rhs_type}")

                # Update symbol table
                actual_type = rhs_type
                if isinstance(rhs_type, DictType) and name in rhs_type.fields:
                    actual_type = rhs_type.fields[name]
                self.symbol_table[name] = actual_type
                if not self.silent:
                    print(f"  [OK] {name} : {actual_type}")

    def infer_type(self, expr, scope) -> Type:
        if isinstance(expr, Literal):
            return expr.type_
        if isinstance(expr, VarRef):
            if expr.name not in scope:
                raise TypeError(f"Undefined variable '{expr.name}'")
            return scope[expr.name]

        if isinstance(expr, Block):
            fields = {k: self.infer_type(v, scope)
                      for k, v in expr.assigns.items()}
            return DictType(fields)

        if isinstance(expr, Call):
            # 1. Retrieve Definition
            if expr.func not in self.func_signatures:
                raise TypeError(f"Unknown function '{expr.func}'")

            defn = self.func_signatures[expr.func]

            # Handle User Defined Macro/Template (Dynamic dispatch)
            if isinstance(defn, FuncDef):
                # Create local scope with argument types passed in
                # This requires re-analyzing the body with concrete types
                # Simplification for demo: return a Generic Signal
                return Generic("Signal", Atomic("Float"))

            # Handle Standard Library (Operator)
            sig = defn  # This is an Operator(Args, Ret)

            # 2. Check Arguments
            arg_types_supplied = {}
            for k, v in expr.args.items():
                arg_types_supplied[k] = self.infer_type(v, scope)

            # 3. Validate Arguments against Operator Signature
            # Case A: Dict args
            if isinstance(sig.args, DictType):
                for req_k, req_t in sig.args.fields.items():
                    if req_k not in arg_types_supplied:
                        raise TypeError(
                            f"Missing argument '{req_k}' in call to '{expr.func}'")
                    if not req_t.matches(arg_types_supplied[req_k]):
                        raise TypeError(
                            f"Argument '{req_k}' type mismatch in '{expr.func}'. Expected {req_t}, got {arg_types_supplied[req_k]}")

            # Case B: Tuple args (Positional)
            elif isinstance(sig.args, TupleType):
                # Convert supplied dict to list based on keys '0', '1'...
                supplied_list = [arg_types_supplied[str(
                    i)] for i in range(len(arg_types_supplied))]
                if len(supplied_list) != len(sig.args.items):
                    raise TypeError(f"Arg count mismatch for '{expr.func}'")
                for expected, actual in zip(sig.args.items, supplied_list):
                    if not expected.matches(actual):
                        raise TypeError(
                            f"Positional arg mismatch. Expected {expected}, got {actual}")

            return sig.ret
