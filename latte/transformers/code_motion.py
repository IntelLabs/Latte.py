import ast
import latte.util as util
import ctree.c.nodes as C


class InvariantLoadStoreLifter(ast.NodeTransformer):
    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        node.body = util.flatten([self.visit(s) for s in node.body])
        if node.init.left.name == "_neuron_index_0":
            # Don't lift out of outer most loop
            return node
        pre_stmts = []
        new_body = []
        post_stmts = []
        loop_var = node.init.left.name
        deps = set()
        for stmt in node.body:
            # print(astor.dump_tree(stmt))
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
            elif isinstance(stmt, C.BinaryOp) and \
                 isinstance(stmt.op, C.Op.Assign) and \
                 isinstance(stmt.left, C.SymbolRef) and \
                 stmt.left.type is not None and \
                    not util.contains_symbol(stmt, loop_var) and \
                    not any(util.contains_symbol(stmt, dep) for dep in deps):
                pre_stmts.append(stmt)
            else:
                new_body.append(stmt)
                if isinstance(stmt, C.BinaryOp) and \
                   isinstance(stmt.op, C.Op.Assign) and \
                   isinstance(stmt.left, C.SymbolRef) and \
                   stmt.left.type is not None:
                    deps.add(stmt.left.name)
        node.body = new_body
        return pre_stmts + [node] + post_stmts


def lift_invariant_load_stores(ast):
    return InvariantLoadStoreLifter().visit(ast)
