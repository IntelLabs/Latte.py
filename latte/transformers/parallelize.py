import ast
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate

class LatteRuntimeLoopParallel(ast.NodeTransformer):
    def visit_For(self, node):
        node.body = [self.visit(s) for s in node.body]
        if hasattr(node, 'parallel') and node.parallel:
            return StringTemplate("""
                parallel_for(0,$looplen / $loopincr,
                  [=](int low, int high) {
                    for (int tmp_$loopvar = low; tmp_$loopvar < high; tmp_$loopvar++) {
                      int $loopvar = tmp_$loopvar * $loopincr;
                      $body;
                    }
                  }
                );
                """, {
                    "looplen": node.test.right,
                    "loopvar": node.test.left,
                    "loopincr": node.incr.value,
                    "body": node.body
                })
        return node

class LatteOpenMPParallel(ast.NodeTransformer):
    def visit_For(self, node):
        if hasattr(node, 'parallel') and node.parallel:
            node.pragma = "omp parallel for"
            # Supports depth one nesting with collapse
            if len(node.body) == 1 and hasattr(node.body[0], 'parallel') and \
                    node.body[0].parallel:
                node.pragma += " collapse(2)"
        return node


LATTE_PARALLEL_MODE = "SIMPLE_LOOP"

def parallelize(tree):
    if LATTE_PARALLEL_MODE == "SIMPLE_LOOP":
        return LatteRuntimeLoopParallel().visit(tree)
    elif LATTE_PARALLEL_MODE == "FLOWGRAPH_LOOP":
        raise NotImplementedError()
    elif LATTE_PARALLEL_MODE == "OPENMP":
        return LatteOpenMPParallel().visit(tree)
    else:
        raise NotImplementedError()
