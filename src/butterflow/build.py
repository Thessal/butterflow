from butterflow.parser import FuncDef, Assign, Literal, Stmt, VarRef, Block, Call
from typing import List
from butterflow.operators import STD_LIB, STD_LIB_IMPL

# ==========================================
# PHASE 4: Graph Builder (Python Classes)
# ==========================================

DEBUG = False


def dump_statements(stmts: List[Stmt]):
    if DEBUG:
        # repr impl for List[Stmt]
        print("DEBUG - Statement sequence dump")
        for x in stmts:
            print(repr(x.__dict__))
            if isinstance(x, FuncDef):
                for y in x.body.assigns:
                    print("  ", y.__dict__)
        print("===")


def dump_scope(scope):
    if DEBUG:
        print("DEBUG - scope", scope)


class Builder:
    def __init__(self):
        self.scope = {}
        self.macros = {}
        # std_lib_nodes should be a dict of constructors, e.g., {'ADD': AddNode, ...}
        # self.std_lib = std_lib_nodes

    def build(self, stmts):
        print("\nBuilding Graph...")
        dump_statements(stmts)
        for stmt in stmts:
            if isinstance(stmt, FuncDef):
                # 1. Store the macro definition for later expansion
                self.macros[stmt.name] = stmt
            elif isinstance(stmt, Assign):
                # 2. Evaluate the RHS to get a Node (not a value)
                res = self.eval(stmt.expr, self.scope)

                # 3. Store the Node in the scope
                assign_name, assign_type = stmt.target
                self.scope[assign_name] = res

        return self.scope.get('result')

    def eval(self, expr, scope):
        # Base Case: Return a literal Node (or raw value if you prefer literals to be raw)
        if isinstance(expr, Literal):
            return expr.val

        # Base Case: Look up the Node associated with this variable name
        if isinstance(expr, VarRef):
            dump_scope(scope)
            if expr.name not in scope:
                raise ValueError(f"Undefined variable: {expr.name}")
            return scope[expr.name]

        # Recursive Case: Evaluate Block
        if isinstance(expr, Block):
            last_val = None
            for assign in expr.assigns:
                val = self.eval(assign.expr, scope)
                scope[assign.target[0]] = val  # Update scope immediately
                last_val = val
            return last_val

        # Core Logic: Function Call / Macro Expansion
        if isinstance(expr, Call):
            # 1. Recursively build the input graphs for arguments first
            # This ensures 'args' contains NODES, not values.
            evaluated_args = {k: self.eval(v, scope)
                              for k, v in expr.args.items()}

            # CASE A: Macro Expansion (Inlining)
            if expr.func in self.macros:
                macro = self.macros[expr.func]

                # Scope Hygiene
                # Do NOT use scope.copy(). Start fresh.
                # Only explicitly passed arguments should exist in the macro scope.
                local_scope = {}

                # Map arguments to the local scope
                # Handles both named args (dict keys) and positional args (indices mapped to names)
                for i, arg_name in enumerate(macro.args):
                    if arg_name in evaluated_args:
                        local_scope[arg_name] = evaluated_args[arg_name]
                    elif str(i) in evaluated_args:
                        local_scope[arg_name] = evaluated_args[str(i)]
                    else:
                        raise ValueError(
                            f"Missing argument '{arg_name}' for macro '{expr.func}'")

                # Recurse: Build the graph INSIDE the macro using the input nodes
                return self.eval(macro.body, local_scope)

            # CASE B: Standard Library Node Creation
            if expr.func in STD_LIB:
                # Instead of executing math, we instantiate a Node.
                # The arguments passed are the Upstream Nodes.
                node_factory = STD_LIB_IMPL.get(expr.func)
                return node_factory(**evaluated_args)

        return None
