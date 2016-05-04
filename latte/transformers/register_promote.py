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
        if "_mm" in node.func.name and "_load_" in node.func.name:
            if node.codegen() not in self.loads:
                self.loads[node.codegen()] = [node, 0]
            self.loads[node.codegen()][1] += 1
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

class VectorLoadRegisterPromoter(ast.NodeTransformer):
    _tmp = -1
    def _gen_register(self):
        self._tmp += 1
        return "___x" + str(self._tmp)

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            collector = VectorLoadCollector()
            [collector.visit(s) for s in node.body]
            for stmt in  collector.loads.keys():
                load_node, number = collector.loads[stmt]
                if number > 1:
                    reg = self._gen_register()
                    replacer = VectorLoadReplacer(stmt, C.SymbolRef(reg))
                    node.body = [replacer.visit(s) for s in node.body]
                    node.body.insert(0, C.Assign(C.SymbolRef(reg, ctree.simd.types.m256()), load_node))
            node.body = util.flatten(node.body)
        return node

def register_promote_vector_loads(ast):
    return VectorLoadRegisterPromoter().visit(ast)
