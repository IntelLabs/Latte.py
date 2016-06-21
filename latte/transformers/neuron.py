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

            ndim = self.ensemble.ndim
            offset = 0

            if node.attr.endswith("input"):
                assert isinstance(self.ensemble, latte.ensemble.ActivationEnsemble)
                # ActivationEnsembles support the self.input construct that is
                # equivalent to self.inputs[neuron_index...][0]
                name += "s"
                self.seen_vars.add(name)
                args = [ast.Name("_neuron_index_{}".format(i), ast.Load()) for i in range(ndim + 1)]
                field = name.replace(self.ensemble.name, "")
                if field in self.ensemble.tiling_info:
                    for dim, _ in self.ensemble.tiling_info[field]:
                        dim += 1  # offset for batch dimension
                        args.append(ast.Name("_neuron_index_{}_inner".format(dim), ast.Load()))
                # Tile 1st (non-batch) dimension
                if False:
                    args[1].id += "_outer"
                    args.append(ast.Name("_neuron_index_1_inner", ast.Load()))
                for i, p in enumerate(self.ensemble.pad):
                    if p > 0:
                        args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p))
                return ast.Subscript(ast.Name(name, ast.Load()), 
                                     ast.Index(ast.Tuple(args, ast.Load())), node.ctx)

            # mark as seen
            self.seen_vars.add(name)

            if node.attr in ["inputs", "grad_inputs"]:
                # only generate batch index for inputs/grad_inputs because
                # the user will provide rest of indices in expression
                ndim = 1
            elif node.attr in self.ensemble.batch_fields:
                # increment ndim for fields that have a batch dimension
                ndim += 1
            else:
                # fields that don't have a batch dimension start at an offset 1
                # as 0 is the batch dimension
                offset = 1

            args = []

            if "grad_" in node.attr and not node.attr.endswith("inputs"):
                # We privatize these buffers and reduce across threads at the
                # end, removing need for synchronization.  This is done by
                # adding an outer dimension of size num_threads to the buffer
                if True:
                    # args.append(ast.Call(ast.Name("omp_get_thread_num", ast.Load()), [], []))
                    args.append(ast.Name("_neuron_index_0", ast.Load()))

            # only append dimensions if it is not fixed in self.buffer_dim_info
            # (used for values shared across a dimension)
            for i in range(ndim):
                if name not in self.buffer_dim_info or not self.buffer_dim_info[name][i]:
                    args.append(ast.Name("_neuron_index_{}".format(i + offset), ast.Load()))
                    # if i + offset == 1:
                    #     args[-1].id += "_outer"

            if node.attr in ["value", "grad"]:
                if node.attr in self.ensemble.tiling_info:
                    for dim, _ in self.ensemble.tiling_info[node.attr]:
                        dim += 1  # offset for batch dimension
                        args.append(ast.Name("_neuron_index_{}_inner".format(dim), ast.Load()))
                for i, p in enumerate(self.ensemble.pad):
                    if p > 0:
                        args[i + 1] = ast.BinOp(args[i + 1], ast.Add(), ast.Num(p))

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

            field = value.value.id.replace(self.ensemble.name, '')
            if field in self.ensemble.tiling_info:
                for dim, _ in self.ensemble.tiling_info[field]:
                    # dim += 1  # offset for batch dimension
                    if "grad_" in field and field != "grad_inputs":
                        dim += 1  # offset for omp_get_thread_num()
                    elif field in self.ensemble.batch_fields:
                        dim += 1
                    index = value.slice.value.elts[dim]
                    if isinstance(index, ast.Name):
                        orig_var = index.id
                        value.slice.value.elts.append(ast.Name(orig_var + "_inner", ast.Load()))
                    elif isinstance(value.slice.value.elts[dim], ast.Num) and \
                            index.n == 0:
                        value.slice.value.elts.append(
                            ast.Name("_input_offset_{}_inner".format(dim), ast.Load()))
                    else:
                        raise NotImplementedError(type(value.slice.value.elts[dim]))
            if "inputs" in value.value.id or "grad_inputs" in value.value.id:
                # Add the input offsets defined by user's mapping for the
                # connection
                ndim = self.ensemble.ndim
                # if isinstance(value.slice.value.elts[1], ast.Num) and value.slice.value.elts[1].n == 0:
                #     value.slice.value.elts.append(ast.Name("_input_offset_1_inner", ast.Load()))
            
                value.slice.value.elts[1:ndim + 1] = [
                    ast.BinOp(value, ast.Add(), 
                        ast.Name("_input_offset_{}".format(i + 1), ast.Load())) 
                    for i, value in enumerate(value.slice.value.elts[1:ndim + 1])
                ]
            # return child node
            return value
        else:
            raise NotImplementedError()
        return node

    counter = -1
    def _gen_unique_variable(self):
        NeuronTransformer.counter += 1
        return "__unique_loopvar{}".format(NeuronTransformer.counter)

    def visit_For(self, node):
        """
        Converts iteration expressions into RangeDim semantic nodes
        """
        _range = node.iter
        if isinstance(_range, ast.Call) and _range.func.id == "eachindex":
            loopvars = []
            for dim in self.connections[0].mapping.shape:
                loopvars.append(self._gen_unique_variable())
            nodes = []
            for index, var in enumerate(loopvars):
                nodes.append(ast.For(
                    ast.Name(var, ast.Store()),
                    ast.Call(ast.Name("range_dim", ast.Load()), [_range.args[0], ast.Num(index)], []),
                    [], []
                ))
            index_expr = ast.Tuple([ast.Name(var, ast.Load()) for var in loopvars], ast.Load())
            nodes[-1].body = [util.replace_name(node.target, index_expr, s) for s in node.body]
            for i in reversed(range(1, len(nodes))):
                nodes[i - 1].body.append(nodes[i])
            return self.visit(nodes[0])
        elif isinstance(_range, ast.Call) and _range.func.id in ["enumerate_dim", "range_dim"]:
            node.body = [self.visit(s) for s in node.body]
            node.body = util.flatten(node.body)
            return RangeDim(node, self.connections[0].mapping, self.connections[0].source)
        else:
            raise NotImplementedError(ast.dump(node))
