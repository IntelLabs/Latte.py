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
    for statement in statements:
        for dep in deps:
            if dep in util.collect_stores(statement):
                dep_statements.append(statement)
                deps = deps.union(util.collect_loads(statement))
    return dep_statements
            

class ConvertEnumerateRange(ast.NodeTransformer):
    """
    converts for ... in enumerate(range(...)) into a valid C for loop
    """
    def __init__(self, direction):
        super().__init__()
        self.blocked_loops = []
        self.tiled_buffers = {}
        self.direction = direction

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
                if len(self.blocked_loops) > 0:
                    curr_loop = self.blocked_loops[0]
                    new_body.append(curr_loop)
                    for loop in self.blocked_loops[1:]:
                        curr_loop.body = [loop]
                        curr_loop = loop
                    curr_loop.body = [result]
                    self.blocked_loops = []
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
        # mapping_func = util.get_ast(node.mapping).body[0]
        ensemble = node.ensemble
        ndim = node.mapping.ndim
        dim = iter.args[1].n
        length = len(node.mapping.shape[dim])
        if isinstance(iter, ast.Call) and iter.func.id == "enumerate_dim":
            raise NotImplementedError()
            # # grab closure variables and inline them into the mapping ast
            # closure_vars = inspect.getclosurevars(node.mapping)
            # for var, value in closure_vars.nonlocals.items():
            #     mapping_func = util.inline_variable(var, value, mapping_func)

            # # replace argument variables with loop variables corresponding to 
            # # the current _neuron_index
            # for i, arg in enumerate(mapping_func.args.args):
            #     i += 1  # offset batch
            #     mapping_func = util.inline_variable(arg.arg, "_neuron_index_{}".format(i), mapping_func)

            # range_expr = mapping_func.body[-1].value.elts[dim]
            # if len(range_expr.args) == 2:
            #     offset = range_expr.args[0]
            # elif len(range_expr.args) == 3:
            #     raise NotImplementedError()
            # else:
            #     offset = 0

            # enum_var, loop_var = node.child_for.target.elts
            # if offset == 0:
            #     # body = [C.Assign(C.SymbolRef(loop_var.id, ctypes.c_int()), 
            #     #                  C.SymbolRef(enum_var.id))]
            #     body = []
            #     node.body = [util.replace_name(loop_var, enum_var, s) for s in node.child_for.body]
            # else:
            #     body = get_dependent_statements(mapping_func.body[:-1], offset.id)
            #     for stmt in body:
            #         # Assume everything in mapping is an int
            #         # FIXME: Do we need to support other kinds of expressions?
            #         stmt.targets[0] = C.SymbolRef(stmt.targets[0].id, ctypes.c_int())
            #     body.append(C.Assign(C.SymbolRef(loop_var.id, ctypes.c_int()), 
            #                          C.Add(C.SymbolRef(enum_var.id), C.Constant(offset))))
            # if dim == 0 and isinstance(iter.args[0], ast.Attribute) and iter.args[0].attr == "inputs":
            #     self.blocked_loops.append(
            #         C.For(
            #             C.Assign(C.SymbolRef(enum_var.id + "_tile", ctypes.c_int()), C.Constant(0)),
            #             C.Lt(C.SymbolRef(enum_var.id + "_tile"), C.Constant(length // latte.core.TILE_SIZE)),
            #             C.PostInc(C.SymbolRef(enum_var.id + "_tile")),
            #             [])
            #     )
            #     if length % latte.core.TILE_SIZE == 0:
            #         length = C.SymbolRef("TILE_SIZE")
            #     else:
            #         raise NotImplementedError()
            #     new_body = []
            #     for statement in node.child_for.body:
            #         result, tiled_buffers = util.tile_array_refs(enum_var.id, statement)
            #         new_body.append(result)
            #         self.tiled_buffers = dict(self.tiled_buffers, **tiled_buffers)
            #     node.child_for.body = new_body
            # body += [self.visit(s) for s in node.child_for.body]

            # return C.For(
            #     C.Assign(C.SymbolRef(enum_var.id, ctypes.c_int()), C.Constant(0)),
            #     C.Lt(C.SymbolRef(enum_var.id), C.Constant(length)),
            #     C.PostInc(C.SymbolRef(enum_var.id)),
            #     body,
            #     "unroll({})".format(length)
            # )
        elif isinstance(iter, ast.Call) and iter.func.id == "range_dim":
            loop_var = node.child_for.target.id
            if dim == 0:
                # self.blocked_loops.append(
                #     C.For(
                #         C.Assign(C.SymbolRef(loop_var + "_tile", ctypes.c_int()), C.Constant(0)),
                #         C.Lt(C.SymbolRef(loop_var + "_tile"), C.Constant(length // latte.core.TILE_SIZE)),
                #         C.PostInc(C.SymbolRef(loop_var + "_tile")),
                #         [])
                # )
                # if length % latte.core.TILE_SIZE == 0:
                #     length = latte.core.TILE_SIZE
                # else:
                #     raise NotImplementedError()
                new_body = []
                for statement in node.child_for.body:
                    result, tiled_buffers = util.tile_array_refs(loop_var, statement)
                    new_body.append(result)
                    self.tiled_buffers = dict(self.tiled_buffers, **tiled_buffers)
                node.child_for.body = new_body

            offset = node.mapping.get_offset(dim)

            body = []
            if not (isinstance(offset, ast.Num) and offset.n == 0):
                def gen_clamp(index):
                    return C.FunctionCall(C.SymbolRef("MIN"), 
                        [C.FunctionCall(C.SymbolRef("MAX"), 
                            [index,
                             C.Constant(0)]), 
                         C.Constant(ensemble.shape[dim] - 1)])
                node.child_for.body = [ClampInputIndex(loop_var, gen_clamp).visit(s) for s in node.child_for.body]
                if dim == 0:
                    node.child_for.body = [ClampInputIndex(loop_var + "_inner", gen_clamp).visit(s) for s in node.child_for.body]
            body += [self.visit(s) for s in node.child_for.body]
            if dim == 0:
                # body = [UpdateInputIndices(input_offset + "_inner_index", input_offset + "_inner").visit(s) for s in body]
                # body.insert(0, (
                #     C.Assign(C.SymbolRef(input_offset + "_inner_index", ctypes.c_int()),
                #              C.Add(C.SymbolRef(loop_var + "_inner"), C.SymbolRef(input_offset + "_inner")))))
                self.blocked_loops.append(C.For(
                    C.Assign(C.SymbolRef(loop_var, ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var), C.Constant(length // latte.core.SIMDWIDTH)),
                    C.PostInc(C.SymbolRef(loop_var)),
                    [],
                    # "unroll_and_jam"
                ))
                return C.For(
                    C.Assign(C.SymbolRef(loop_var + "_inner", ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var + "_inner"), C.Constant(latte.core.SIMDWIDTH)),
                    C.PostInc(C.SymbolRef(loop_var + "_inner")),
                    body,
                    # "unroll_and_jam({})".format(latte.core.SIMDWIDTH)
                    # "unroll({})".format(latte.core.SIMDWIDTH)
                    "unroll"
                )
            else:
                return C.For(
                    C.Assign(C.SymbolRef(loop_var, ctypes.c_int()), C.Constant(0)),
                    C.Lt(C.SymbolRef(loop_var), C.Constant(length)),
                    C.PostInc(C.SymbolRef(loop_var)),
                    body,
                    # "unroll({})".format(length)
                    "unroll"
                )
        raise NotImplementedError()

def convert_enumerate_ranges(ast, direction):
    visitor = ConvertEnumerateRange(direction)
    ast = visitor.visit(ast)
    return ast, visitor.tiled_buffers

class ClampInputIndex(ast.NodeTransformer):
    def __init__(self, loop_var, gen_clamp):
        self.loop_var = loop_var
        self.gen_clamp = gen_clamp

    def visit_BinaryOp(self, node):
        if isinstance(node.op, C.Op.ArrayRef):
            curr_node = node
            while not isinstance(curr_node, ast.Name):
                curr_node = curr_node.left
            if curr_node.id.endswith("inputs"):
                curr_node = node
                while not util.contains_name(curr_node.right, self.loop_var):
                    curr_node = curr_node.left
                curr_node.right = self.gen_clamp(curr_node.right)
                return node
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        return node

