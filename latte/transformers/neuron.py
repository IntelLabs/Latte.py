import ast
import inspect
import latte.util as util
import latte
import ctree.c.nodes as C
import ctypes

class RangeDim(ast.AST):
    _fields = ['child_for']

    def __init__(self, child_for, mapping, ensemble):
        super().__init__()
        self.child_for = child_for
        self.mapping = mapping
        self.ensemble = ensemble


class NeuronTransformer(ast.NodeTransformer):
    def __init__(self, ensemble, connections, buffer_dim_info):
        super().__init__()
        self.ensemble = ensemble
        self.seen_vars = set()
        self.connections = connections
        self.buffer_dim_info = buffer_dim_info

    def visit(self, node):
        """
        Support replacing nodes with a list of nodes by flattening `body`
        fields.
        """
        node = super().visit(node)
        if hasattr(node, 'body'):
            node.body = util.flatten(node.body)
        return node

    def visit_Attribute(self, node):
        """
        A reference `self.field[...]` will be replaced with an array reference
        ensemble_namefield[...] to reflect SOA (struct of array) layout.
        """
        if node.value.id == "self":
            # name is ensemble name + attribute
            name = self.ensemble.name + node.attr

            # mark as seen
            self.seen_vars.add(name)

            ndim = self.ensemble.ndim
            offset = 0

            if node.attr in self.ensemble.batch_fields:
                # increment ndim for fields that have a batch dimension
                ndim += 1
            elif node.attr in ["inputs", "grad_inputs"]:
                if isinstance(self.ensemble, latte.ensemble.ActivationEnsemble):
                    ndim += 1
                else:
                    # only generate batch index for inputs/grad_inputs because
                    # the user will provide rest of indices in expression
                    ndim = 1
            else:
                # fields that don't have a batch dimension start at an offset 1
                # as 0 is the batch dimension
                offset = 1
            args = []
            if "grad_" in node.attr and not node.attr.endswith("inputs"):
                args.append(ast.Call(ast.Name("omp_get_thread_num", ast.Load()), [], []))
            for i in range(ndim):
                # only append this dimension if it is not fixed in self.buffer_dim_info
                # (used for shared values)
                if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]:
                    args.append(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()))
                    if i + offset == 1:
                        args[-1].id += "_outer"

            if node.attr in ["value", "grad"]:
                args.append(ast.Name("_neuron_index_1_inner", ast.Load()))

            # return updated indedxing expression
            return ast.Subscript(ast.Name(name, ast.Load()), 
                                 ast.Index(ast.Tuple(args, ast.Load())), node.ctx)
        else:
            raise Exception("Unsupported Attribute node")

    def visit_Subscript(self, node):
        """
        If self.visit(node.value) returns a Subscript, flatten that expression
        into the current subscript node.
        """
        node.value = self.visit(node.value)
        if isinstance(node.value, ast.Subscript):
            value = node.value
            # append or extend the indexing expressions for *current* node to child node
            if isinstance(node.slice.value, (ast.Name, ast.Num)):
                value.slice.value.elts.append(node.slice.value)
            elif isinstance(node.slice.value, ast.Tuple):
                value.slice.value.elts.extend(node.slice.value.elts)
            else:
                raise NotImplementedError(node.slice.value)
            # return child node
            if "inputs" in value.value.id or "grad_inputs" in value.value.id:
                ndim = self.ensemble.ndim
                value.slice.value.elts[1:ndim + 1] = [ast.Name("_input_offset_{}".format(i + 1), ast.Load()) for i in range(len(value.slice.value.elts[1:ndim + 1]))]
                value.slice.value.elts.append(ast.Name("_input_offset_1_inner", ast.Load()))
            else:
                value.slice.value.elts.append(ast.Name("_neuron_index_1_inner", ast.Load()))
            return value
        else:
            raise NotImplementedError()
        return node

    def visit_For(self, node):
        """
        Converts iteration expressions into enumerate and range calls
        """
        _range = node.iter
        if isinstance(_range, ast.Call) and _range.func.id in ["enumerate_dim", "range_dim"]:
            node.body = [self.visit(s) for s in node.body]
            node.body = util.flatten(node.body)
            return RangeDim(node, self.connections[0].mapping, self.connections[0].source)
            # grab closure variables and inline them into the mapping ast
            closure_vars = inspect.getclosurevars(self.connections[0].mapping)
            mapping_func = util.get_ast(self.connections[0].mapping).body[0]
            for var, value in closure_vars.nonlocals.items():
                mapping_func = util.inline_variable(var, value, mapping_func)

            # replace argument variables with loop variables corresponding to 
            # the current _neuron_index
            for i, arg in enumerate(mapping_func.args.args):
                i += 1  # offset batch
                mapping_func = util.inline_variable(arg.arg, "_neuron_index_{}".format(i), mapping_func)
            
            # the dimension is the second argument i.e. range_dim(self.inputs, dim)
            target_dim = _range.args[1].n
            shape = mapping_func.body[-1].value
            node.iter = shape.elts[target_dim]
            # assert node.iter.func.id == "range"
            # if len(node.iter.args) == 1:
            #     if isinstance(node.iter.args[0], ast.Name):
            #         node.iter.args[0] = ast.Call(ast.Name("MIN", ast.Load()), 
            #                 [node.iter.args[0], C.Constant(self.ensemble.shape[target_dim])], [])
            # elif len(node.iter.args) == 2:
            #     if isinstance(node.iter.args[0], ast.Name):
            #         node.iter.args[0] = ast.Call(ast.Name("MAX", ast.Load()), [node.iter.args[0], C.Constant(0)], [])
            #     if isinstance(node.iter.args[1], ast.Name):
            #         node.iter.args[1] = ast.Call(ast.Name("MIN", ast.Load()), 
            #                 [node.iter.args[1], C.Constant(self.ensemble.shape[target_dim])], [])
            # else:
            #     raise NotImplementedError()


            if _range.func.id == "enumerate_dim":
                node.iter = ast.Call(ast.Name("enumerate", ast.Load()), [node.iter], [])

            if self.connections[0].mapping_inserted:
                pre_stmts = []
            else:
                pre_stmts = mapping_func.body[:-1]
                for stmt in pre_stmts:
                    assert isinstance(stmt, ast.Assign)
                    assert len(stmt.targets) == 1
                    stmt.targets[0] = C.SymbolRef(stmt.targets[0].id, ctypes.c_int())
                self.connections[0].mapping_inserted = True

            node.body = [self.visit(s) for s in node.body]
            node.body = util.flatten(node.body)
            return pre_stmts + [node]
        else:
            raise NotImplementedError(ast.dump(node))
