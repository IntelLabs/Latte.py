import ast
import ctree.c.nodes as C
from ctree.templates.nodes import StringTemplate
import latte.config

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
            elif all(isinstance(s, C.For) and hasattr(s, 'parallel') and s.parallel for s in node.body):
                # FIXME: Distribute the loops and use collapse
                raise NotImplementedError()
        return node


def parallelize(tree):
    if latte.config.parallel_strategy == "SIMPLE_LOOP":
        return LatteRuntimeLoopParallel().visit(tree)
    elif latte.config.parallel_strategy == "FLOWGRAPH_LOOP":
        raise NotImplementedError()
    elif latte.config.parallel_strategy == "OPENMP":
        return LatteOpenMPParallel().visit(tree)
    else:
        raise NotImplementedError()
