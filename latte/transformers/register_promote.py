import ast
import ctree.c.nodes as C
import ctree
import latte.util as util

class VectorLoadCollector(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.loads = {}

    def visit(self, node):
        # Don't descend into nested expressions
        if hasattr(node, 'body'):
            return
        super().visit(node)

    def visit_FunctionCall(self, node):
        if "_mm" in node.func.name and ("_load_" in node.func.name or "_set1" in node.func.name):
            if node.codegen() not in self.loads:
                self.loads[node.args[0].codegen()] = [node.args[0], 0, node.func.name]
            self.loads[node.args[0].codegen()][1] += 1
        [self.visit(arg) for arg in node.args]

class VectorLoadReplacer(ast.NodeTransformer):
    def __init__(self, load_stmt, new_stmt):
        self.load_stmt = load_stmt
        self.new_stmt = new_stmt

    def visit_FunctionCall(self, node):
        if node.codegen() == self.load_stmt:
            return self.new_stmt
        node.args = [self.visit(arg) for arg in node.args]
        return node

class VectorLoadStoresRegisterPromoter(ast.NodeTransformer):
    _tmp = -1
    def _gen_register(self):
        VectorLoadStoresRegisterPromoter._tmp += 1
        return "___x" + str(self._tmp)

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            # [collector.visit(s) for s in node.body]
            new_body = []
            seen = {}
            stores = []
            collector = VectorLoadCollector()
            for s in node.body:
                collector.visit(s)
                for stmt in collector.loads.keys():
                    if stmt not in seen:
                        reg = self._gen_register()
                        load_node, number, func = collector.loads[stmt]
                        seen[stmt] = (reg, load_node, func)
                        new_body.append(C.Assign(C.SymbolRef(reg, ctree.simd.types.m256()),
                                                 C.FunctionCall(C.SymbolRef(func), [load_node])))
                if isinstance(s, C.FunctionCall) and "_mm" in s.func.name and "_store" in s.func.name:
                    if s.args[0].codegen() in seen:
                        stores.append((s.args[0], seen[s.args[0].codegen()][0]))
                        s = C.Assign(C.SymbolRef(seen[s.args[0].codegen()][0]), s.args[1])
                for stmt in seen.keys():
                    reg, load_node, func = seen[stmt]
                    replacer = VectorLoadReplacer(
                            C.FunctionCall(C.SymbolRef(func), [load_node]).codegen(), 
                            C.SymbolRef(reg))
                    s = replacer.visit(s)
                new_body.append(s)
            for target, value in stores:
                new_body.append(C.FunctionCall(C.SymbolRef("_mm256_store_ps"), [target, C.SymbolRef(value)]))
            node.body = util.flatten(new_body)
        return node

def register_promote_vector_loads_stores(ast):
    return VectorLoadStoresRegisterPromoter().visit(ast)


class InvariantLoadStoreLifter(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        pre_stmts = []
        new_body = []
        post_stmts = []
        loop_var = node.init.left.name
        deps = set()
        for stmt in node.body:
            if isinstance(stmt, C.FunctionCall) and "_mm" in stmt.func.name and \
                "_store" in stmt.func.name and \
                not util.contains_symbol(stmt, loop_var) and \
                not any(util.contains_symbol(stmt, dep) for dep in deps):
                    post_stmts.append(stmt)
            elif isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and \
                    isinstance(stmt.right, C.FunctionCall) and "_load" in stmt.right.func.name and \
                    not util.contains_symbol(stmt, loop_var) and \
                    not any(util.contains_symbol(stmt, dep) for dep in deps):
                pre_stmts.append(stmt)
            elif isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and stmt.left.type is not None and \
                    not util.contains_symbol(stmt, loop_var) and \
                    not any(util.contains_symbol(stmt, dep) for dep in deps):
                pre_stmts.append(stmt)
            else:
                new_body.append(stmt)
                if isinstance(stmt, C.BinaryOp) and isinstance(stmt.op, C.Op.Assign) and stmt.left.type is not None:
                    deps.add(stmt.left.name)
        node.body = new_body
        return pre_stmts + [node] + post_stmts


def lift_invariant_load_stores(ast):
    return InvariantLoadStoreLifter().visit(ast)
