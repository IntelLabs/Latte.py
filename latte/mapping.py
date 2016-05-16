import latte.util as util
import inspect
import ast

class Mapping:
    def __init__(self, mapping_func):
        self.mapping_func = mapping_func
        ast = util.get_ast(mapping_func).body[0]

        closure_vars = inspect.getclosurevars(mapping_func)
        for var, value in closure_vars.nonlocals.items():
            ast = util.inline_variable(var, value, ast)
        self.ast = ast
        self.ndim = len(self.ast.args.args)
        self.shape = mapping_func(*[1 for _ in range(self.ndim)])

    def get_offset(self, dim):
        if self.mapping_func == one_to_one:
            # return ast.Name("_neuron_index_{}".format(dim + 1), ast.Load())
            return ast.Num(0)
        range_expr = self.ast.body[-1].value.elts[dim]
        if len(range_expr.args) == 2:
            return range_expr.args[0]
        elif len(range_expr.args) == 3:
            raise NotImplementedError()
        else:
            return ast.Num(0)

    def is_one_to_one(self):
        return self.mapping_func == one_to_one

def one_to_one(*args):
    return tuple(range(a,a+1) for a in args)
