import ast
import ctree.c.nodes as C
import latte.util as util
import ctypes
import latte.core
import inspect
from ctree.transformations import PyBasicConversions

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
    def __init__(self):
        super().__init__()
        self.blocked_loops = []
        self.tiled_buffers = {}

    def visit(self, node):
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_For(self, node):
        if isinstance(node.iter, ast.Call) and node.iter.func.id == "range" and \
            node.target.id == "_neuron_index_1":
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

    def visit_RangeDim(self, node):
        iter = node.child_for.iter
        mapping_func = util.get_ast(node.mapping).body[0]
        ndim = len(mapping_func.args.args)
        dim = iter.args[1].n
        length = len(node.mapping(*[1 for _ in range(ndim)])[dim])
        if isinstance(iter, ast.Call) and iter.func.id == "enumerate_dim":
            # grab closure variables and inline them into the mapping ast
            closure_vars = inspect.getclosurevars(node.mapping)
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.inline_variable(var, value, mapping_func)

            # replace argument variables with loop variables corresponding to 
            # the current _neuron_index
            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.inline_variable(arg.arg, "_neuron_index_{}".format(i), mapping_func)

            range_expr = mapping_func.body[-1].value.elts[dim]
            if len(range_expr.args) == 2:
                offset = range_expr.args[0]
            elif len(range_expr.args) == 3:
                raise NotImplementedError()
            else:
                offset = 0

            enum_var, loop_var = node.child_for.target.elts
            if offset == 0:
                # body = [C.Assign(C.SymbolRef(loop_var.id, ctypes.c_int()), 
                #                  C.SymbolRef(enum_var.id))]
                body = []
                node.body = [util.replace_name(loop_var, enum_var, s) for s in node.child_for.body]
            else:
                body = get_dependent_statements(mapping_func.body[:-1], offset.id)
                for stmt in body:
                    # Assume everything in mapping is an int
                    # FIXME: Do we need to support other kinds of expressions?
                    stmt.targets[0] = C.SymbolRef(stmt.targets[0].id, ctypes.c_int())
                body.append(C.Assign(C.SymbolRef(loop_var.id, ctypes.c_int()), 
                                     C.Add(C.SymbolRef(enum_var.id), C.Constant(offset))))
            if dim == 0:
                self.blocked_loops.append(
                    C.For(
                        C.Assign(C.SymbolRef(enum_var.id + "_tile", ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef(enum_var.id + "_tile"), C.Constant(length // latte.core.TILE_SIZE)),
                        C.PostInc(C.SymbolRef(enum_var.id + "_tile")),
                        [])
                )
                if length % latte.core.TILE_SIZE == 0:
                    length = C.SymbolRef("TILE_SIZE")
                else:
                    raise NotImplementedError()
                new_body = []
                for statement in node.child_for.body:
                    result, tiled_buffers = util.tile_array_refs(enum_var.id, statement)
                    new_body.append(result)
                    self.tiled_buffers = dict(self.tiled_buffers, **tiled_buffers)
                node.child_for.body = new_body
            body += [self.visit(s) for s in node.child_for.body]
            return C.For(
                C.Assign(C.SymbolRef(enum_var.id, ctypes.c_int()), C.Constant(0)),
                C.Lt(C.SymbolRef(enum_var.id), C.Constant(length)),
                C.PostInc(C.SymbolRef(enum_var.id)),
                body,
                "unroll({})".format(length)
            )
        elif isinstance(iter, ast.Call) and iter.func.id == "range_dim":
            loop_var = node.child_for.target.id
            body = []
            if dim == 0:
                self.blocked_loops.append(
                    C.For(
                        C.Assign(C.SymbolRef(loop_var + "_tile", ctypes.c_int()), C.Constant(0)),
                        C.Lt(C.SymbolRef(loop_var + "_tile"), C.Constant(length // latte.core.TILE_SIZE)),
                        C.PostInc(C.SymbolRef(loop_var + "_tile")),
                        [])
                )
                if length % latte.core.TILE_SIZE == 0:
                    length = C.SymbolRef("TILE_SIZE")
                else:
                    raise NotImplementedError()
                new_body = []
                for statement in node.child_for.body:
                    result, tiled_buffers = util.tile_array_refs(loop_var, statement)
                    new_body.append(result)
                    self.tiled_buffers = dict(self.tiled_buffers, **tiled_buffers)
                node.child_for.body = new_body
            body += [self.visit(s) for s in node.child_for.body]
            return C.For(
                C.Assign(C.SymbolRef(loop_var, ctypes.c_int()), C.Constant(0)),
                C.Lt(C.SymbolRef(loop_var), C.Constant(length)),
                C.PostInc(C.SymbolRef(loop_var)),
                body,
                "unroll({})".format(length)
            )
        raise NotImplementedError()

def convert_enumerate_ranges(ast):
    visitor = ConvertEnumerateRange()
    ast = visitor.visit(ast)
    return ast, visitor.tiled_buffers
