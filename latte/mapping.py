import latte.util as util
import inspect
import ast
from copy import deepcopy

class Mapping:
    def __init__(self, mapping_func, clamp):
        self.mapping_func = mapping_func
        self.clamp = clamp
        tree = util.get_ast(mapping_func).body[0]

        closure_vars = inspect.getclosurevars(mapping_func)
        for var, value in closure_vars.nonlocals.items():
            value = ast.parse(str(value)).body[0].value
            tree = util.inline_variable(var, value, tree)

        self.ast = tree
        self.ndim = len(self.ast.args.args)
        self.shape = mapping_func(*[1 for _ in range(self.ndim)])

    def set_arg(self, dim, value):
        if self.mapping_func == one_to_one:
           return
        self.ast = util.inline_variable(self.ast.args.args[dim].arg, value, self.ast)

    def get_offset(self, dim):
        if self.mapping_func == one_to_one:
            # return ast.Name("_neuron_index_{}".format(dim + 1), ast.Load())
            return ast.Num(0)
        range_expr = self.ast.body[-1].value.elts[dim]
        if len(range_expr.args) >= 2:
            return range_expr.args[0]
        else:
            return ast.Num(0)

    def get_end(self, dim):
        if self.mapping_func == one_to_one:
            # return ast.Name("_neuron_index_{}".format(dim + 1), ast.Load())
            return ast.Num(1)
        range_expr = self.ast.body[-1].value.elts[dim]
        if len(range_expr.args) >= 2:
            return range_expr.args[1]
        else:
            return range_expr.args[0]

    def get_step(self, dim):
        if self.mapping_func == one_to_one:
            return 1
        range_expr = self.ast.body[-1].value.elts[dim]
        if len(range_expr.args) <= 2:
            return 1
        elif len(range_expr.args) == 3:
            return range_expr.args[2].n
        else:
            raise NotImplementedError()

    def is_one_to_one(self):
        return self.mapping_func == one_to_one

def one_to_one(*args):
    return tuple(range(a,a+1) for a in args)
