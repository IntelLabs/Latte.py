import ast
import ctree.c.nodes as C
import latte.util as util
import ctypes
import latte.core
import inspect
from ctree.transformations import PyBasicConversions
import astor

def get_dependent_statements(statements, target):
    deps = set([target])
    dep_statements = []
    for statement in reversed(statements):
        for dep in deps:
            if dep in util.collect_stores(statement):
                dep_statements.append(statement)
                deps = deps.union(util.collect_loads(statement))
    return dep_statements
            

class ConvertEnumerateRange(ast.NodeTransformer):
    """
    converts for ... in enumerate(range(...)) into a valid C for loop
    """
    def __init__(self, direction, ensemble):
        super().__init__()
        self.tiled_loops = []
        self.direction = direction
        self.ensemble = ensemble

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range" and \
            (self.direction == "forward" and node.target.id == "_neuron_index_1_outer") or \
            (self.direction == "backward" and node.target.id == "_neuron_index_0"):
            new_body = []
            for statement in node.body:
                result = self.visit(statement)
                if len(self.tiled_loops) > 0:
                    curr_loop = self.tiled_loops[0]
                    new_body.append(curr_loop)
                    for loop in self.tiled_loops[1:]:
                        curr_loop.body = [loop]
                        curr_loop = loop
                    curr_loop.body = [result]
                    self.tiled_loops = []
                else:
                    new_body.append(result)
            node.body = new_body
            return node
        node.body = util.flatten([self.visit(s) for s in node.body])
        return node

    _tmp = -1
    def _gen_tmp(self):
        self._tmp += 1
        return "input_offset_" + str(self._tmp)

    def visit_RangeDim(self, node):
        iter = node.child_for.iter
        ensemble = node.ensemble
        ndim = node.mapping.ndim
        dim = iter.args[1].n
        offset = node.mapping.get_offset(dim)
        step = node.mapping.get_step(dim)
        length = len(node.mapping.shape[dim])
        if isinstance(iter, ast.Call) and iter.func.id == "range_dim":
            loop_var = node.child_for.target.id

            body = []
            body += [self.visit(s) for s in node.child_for.body]
            if (self.direction == "forward" and "inputs" in self.ensemble.tiling_info and 
                    any(dim == x[0] for x in self.ensemble.tiling_info["inputs"])) or (
                    self.direction == "backward" and "grad_inputs" in self.ensemble.tiling_info and 
                    any(dim == x[0] for x in self.ensemble.tiling_info["grad_inputs"])):
                outer_loop = C.For(
                    C.Assign(C.SymbolRef(loop_var + "_outer", ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var + "_outer"), C.Constant(length // latte.config.SIMDWIDTH)),
                    C.AddAssign(C.SymbolRef(loop_var + "_outer"), C.Constant(1)),
                    []
                )
                self.tiled_loops.append(outer_loop)
                inner_loop = C.For(
                    C.Assign(C.SymbolRef(loop_var + "_inner", ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var + "_inner"), C.Constant(latte.config.SIMDWIDTH)),
                    C.AddAssign(C.SymbolRef(loop_var + "_inner"), C.Constant(1)),
                    body,
                )
                return inner_loop
            else:
                body = [UpdateInputIndices(loop_var, C.Mul(C.SymbolRef(loop_var), C.Constant(step))).visit(s) for s in body]
                return C.For(
                    C.Assign(C.SymbolRef(loop_var, ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var), C.Constant(length)),
                    C.AddAssign(C.SymbolRef(loop_var), C.Constant(1)),
                    body,
                    # "unroll_and_jam({})".format(length)
                    # "unroll"
                )
        raise NotImplementedError()

def convert_enumerate_ranges(node, direction, ensemble):
    return ConvertEnumerateRange(direction, ensemble).visit(node)


class UpdateInputIndices(ast.NodeTransformer):
    def __init__(self, index, to_replace):
        self.index = index
        self.to_replace = to_replace

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            curr_node = node
            while not isinstance(curr_node, ast.Name):
                curr_node = curr_node.left
            if "inputs" in curr_node.id:
                return util.replace_name(ast.Name(self.index, ast.Load()), self.to_replace, node)
        return node


